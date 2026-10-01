"""Enriching content that already exists: examples and relations.

Everything here is additive and best-effort, which is why it runs AFTER the generation transaction
commits: folding a secondary step into the write would roll back a good entry when it failed.

Two guards on the disambiguation path are load-bearing: candidates are ordered deterministically and
capped, and the model's chosen index is validated against that exact list. Trusting the index would
write a resolution pointing at a sense the model never saw.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from lexi_ai.models import BatchResult, ResolveDecision, SenseView
from lexi_ai.text import sense_content_hash

if TYPE_CHECKING:
    from lexi_ai.db.session import SqlAlchemyUnitOfWork

# A generated word can be the target of many pending edges, so the inbound hook gets
# headroom over the raw word count. The hard ceiling still applies.
_INBOUND_FACTOR = 20


class EnrichmentService:
    """Example augmentation and sense-relation resolution."""

    def __init__(
        self,
        uow_factory: Callable[[], SqlAlchemyUnitOfWork],
        example_generator: Callable[[], object],
        judge_factory: Callable[[], object | None],
        themed_examples: Callable[..., object],
        max_examples_per_call: int,
    ) -> None:
        self._uow = uow_factory
        self._example_generator = example_generator
        self._judge_factory = judge_factory
        self._themed_examples = themed_examples
        self._max_examples = max_examples_per_call

    # --- examples -----------------------------------------------------------

    async def add_examples(
        self, sense_id: int, n: int = 3, theme: str | int | None = None
    ) -> SenseView:
        """Append up to ``n`` fresh examples to one sense.

        The one clean generation gap: an example illustrates a sense rather than asserting a new
        fact about it, so generating more cannot fabricate linguistic content. Existing examples are
        fed back to the model so it avoids repeating them.

        ``n`` is a best-effort maximum, clamped to what the output schema accepts — asking for more
        would guarantee a validation failure and burn the retries.
        """
        if theme is not None:
            return await self._themed_examples(sense_id, n, theme)
        async with self._uow() as uow:
            context = await uow.senses.example_context(sense_id)
        if context is None:
            raise ValueError(f"unknown sense_id: {sense_id}")
        facts, existing = context
        n = min(n, self._max_examples)
        if n > 0:
            batch = await self._example_generator().generate_examples(facts, existing, n)
            async with self._uow() as uow:
                await uow.senses.append_examples(sense_id, batch.examples)
                await uow.commit()
        async with self._uow() as uow:
            return (await uow.entries.sense_views([sense_id]))[0]

    # --- relation resolution ------------------------------------------------

    async def resolve_relations(self, batch_size: int = 20) -> list[BatchResult]:
        """Reconcile one batch of pending sense-relation edges: the manual and backfill entry point,
        for words whose hook was skipped.
        """
        return await self.resolve(batch_size, word_ids=None)

    async def resolve_inbound(self, word_ids: Sequence[int]) -> list[BatchResult]:
        """Resolve edges pointing at words that just became done.

        Every error is swallowed: the generation that triggered this is already committed, so a
        judge outage must degrade to leaving the edges pending.
        """
        if not word_ids:
            return []
        try:
            return await self.resolve(len(word_ids) * _INBOUND_FACTOR, list(word_ids))
        except Exception:  # noqa: BLE001 - inbound resolve is strictly best-effort
            return []

    async def resolve(self, batch_size: int, word_ids: Sequence[int] | None) -> list[BatchResult]:
        """Read the pending queue, judge it in one call, apply the verdicts. With no judge
        configured this returns nothing rather than failing.
        """
        judge = self._judge_factory()
        if judge is None:
            return []
        from lexi_ai.providers.wsd import WSD_BATCH_CEIL, pos_filtered_candidates
        from lexi_ai.schemas import WsdCandidate, WsdTask

        capped = max(1, min(batch_size, WSD_BATCH_CEIL))
        async with self._uow() as uow:
            tasks = await uow.senses.pending_relations(
                capped, word_ids=list(word_ids) if word_ids is not None else None
            )
        if not tasks:
            return []

        # Remember the filtered candidate order per edge: the judge answers with an
        # index into exactly this list, so it is the only way back to the sense.
        candidates_by_edge: dict[int, list] = {}
        judge_tasks: list[WsdTask] = []
        for task in tasks:
            candidates = pos_filtered_candidates(task.source_pos, task.candidates)
            candidates_by_edge[task.edge_id] = candidates
            judge_tasks.append(
                WsdTask(
                    rel_type=task.rel_type,
                    gloss=task.gloss,
                    source_def=task.source_def,
                    candidates=[
                        WsdCandidate(index=position, definition=candidate.definition)
                        for position, candidate in enumerate(candidates)
                    ],
                )
            )

        choices = await judge.judge(judge_tasks)
        decisions = [
            self._decide(task, candidates_by_edge[task.edge_id], choice.chosen_index)
            for task, choice in zip(tasks, choices, strict=True)
        ]
        async with self._uow() as uow:
            outcomes = await uow.senses.apply_resolutions(decisions)
            await uow.commit()
        return [
            BatchResult(key=outcome.edge_id, value=outcome.state)
            if outcome.error is None
            else BatchResult(key=outcome.edge_id, error=outcome.error)
            for outcome in outcomes
        ]

    @staticmethod
    def _decide(task, candidates: list, chosen: int | None) -> ResolveDecision:  # noqa: ANN001
        """Turn one judged answer into a decision, refusing an unusable index.

        An out-of-range or absent index means unresolvable, never a guess: writing a resolution the
        model did not choose would point the edge at the wrong sense.
        """
        if chosen is None or not (0 <= chosen < len(candidates)):
            return ResolveDecision(task.edge_id, None, None)
        target = candidates[chosen]
        return ResolveDecision(task.edge_id, target.sense_id, sense_content_hash(target.definition))
