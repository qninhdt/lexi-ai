"""Enriching content that already exists: examples, vectors, and relations.

Everything here is additive and best-effort by design, which is why it runs AFTER
the generation transaction commits rather than inside it. An embedding failure or a
judge outage must leave published content untouched; folding either into the write
would roll back a good entry because a secondary step failed.

Two guards on the disambiguation path are load-bearing rather than defensive:
candidates are ordered deterministically and capped, and the model's chosen index is
validated against that exact list. Trusting the index would write a resolution
pointing at a sense the model never saw.
"""

from collections.abc import Callable, Sequence

from lexi_ai.domain.errors import SemanticSearchDisabled
from lexi_ai.domain.hashing import sense_content_hash
from lexi_ai.domain.models import ResolveDecision, VectorRecord
from lexi_ai.domain.ports import UnitOfWork, VectorIndex
from lexi_ai.normalize import render
from lexi_ai.read_models import BatchResult, SenseView

# A generated word can be the target of many pending edges, so the inbound hook gets
# headroom over the raw word count. The hard ceiling still applies.
_INBOUND_FACTOR = 20


class EnrichmentService:
    """Example augmentation, embedding backfill, and relation resolution."""

    def __init__(
        self,
        uow_factory: Callable[[], UnitOfWork],
        embedder,  # noqa: ANN001 - the encoder; only model_name and embed are used
        example_generator: Callable[[], object],
        judge_factory: Callable[[], object | None],
        read_senses: Callable[[Sequence[int]], object],
        themed_examples: Callable[..., object],
        max_examples_per_call: int,
        vectors: VectorIndex | None,
    ) -> None:
        self._uow = uow_factory
        self._embedder = embedder
        self._vectors = vectors
        self._example_generator = example_generator
        self._judge_factory = judge_factory
        self._read_senses = read_senses
        self._themed_examples = themed_examples
        self._max_examples = max_examples_per_call

    # --- examples -----------------------------------------------------------

    async def add_examples(
        self, sense_id: int, n: int = 3, theme: str | int | None = None
    ) -> SenseView:
        """Append up to ``n`` fresh examples to one sense.

        This is the one clean generation gap: an example illustrates a sense rather
        than asserting a new fact about it, so generating more cannot fabricate
        linguistic content. Existing examples are never touched, and they are fed
        back to the model so it avoids repeating them.

        ``n`` is a best-effort maximum, clamped to what the output schema accepts —
        asking for more would guarantee a validation failure and burn the retries.
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
        return (await self._read_senses([sense_id]))[0]

    # --- embeddings ---------------------------------------------------------

    async def backfill_embeddings(self, *, limit: int | None = None) -> int:
        """Embed done senses the vector index does not already hold; returns the count.

        This is the reconciliation step the eventually-consistent vector store
        depends on. It fills the gaps left by best-effort generation and by an
        encoder change (a vector tagged with another model is ignored until
        replaced), and it prunes vectors whose sense no longer exists — a delete or
        a regeneration leaves those behind, and they would otherwise consume slots
        in a top-k answer.

        Idempotent: with everything embedded and nothing stale, this returns zero.

        RAISES on encoder or index failure. This is the operation whose entire
        purpose is to make the index correct, so a caller must be able to tell
        "nothing needed doing" from "the index is unreachable"; both would
        otherwise be a return of zero. With semantic search switched off it raises
        ``SemanticSearchDisabled`` for the same reason: reconciling an index that
        does not exist is not a success.
        """
        if self._vectors is None:
            raise SemanticSearchDisabled(
                "backfill_embeddings() needs semantic search enabled: set "
                "LEXI_VECTOR_BACKEND=lancedb (durable) or 'memory' (tests only)"
            )
        await self._prune_orphan_vectors()
        return await self.embed_missing(limit=limit)

    async def embed_missing(
        self, word_ids: Sequence[int] | None = None, limit: int | None = None
    ) -> int:
        """Encode and store vectors for senses the index has no current vector for.

        Raises on encoder or index failure: the extra may be uninstalled, the model
        may fail to load, the device may be out of memory. The generation hook that
        must survive all of that swallows it at ITS call site (see
        ``Lexicon._embed_words``); callers who asked for embedding on purpose get
        the error.

        Returns zero without touching the encoder when semantic search is off. This
        runs on every generation, so "the feature is disabled" must be a cheap
        no-op rather than an exception raised and swallowed once per word — and it
        must not drag in the encoder that a disabled feature never needs.
        """
        if self._vectors is None:
            return 0
        model = self._embedder.model_name
        stored = await self._vectors.ids({"model": model})
        async with self._uow() as uow:
            candidates = await uow.senses.needing_embedding(
                word_ids=list(word_ids) if word_ids is not None else None,
                limit=None,
            )
        pending = [row for row in candidates if str(row.sense_id) not in stored]
        if limit is not None:
            pending = pending[:limit]
        if not pending:
            return 0
        texts = [self._embed_text(row.norm, row.definition) for row in pending]
        vectors = await self._embedder.embed(texts)
        if not vectors:
            return 0
        return await self._vectors.upsert(
            [
                VectorRecord(id=str(row.sense_id), vector=vector, meta={"model": model})
                for row, vector in zip(pending, vectors, strict=True)
            ]
        )

    async def _prune_orphan_vectors(self) -> int:
        """Drop vectors whose sense is gone.

        A delete or a regeneration leaves the old vector behind, and it would
        otherwise consume a slot in a top-k answer. Raises like its only caller,
        ``backfill_embeddings``.
        """
        stored = await self._vectors.ids()
        if not stored:
            return 0
        async with self._uow() as uow:
            live = await uow.senses.live_sense_ids()
        orphans = [
            stored_id
            for stored_id in stored
            if not stored_id.isdigit() or int(stored_id) not in live
        ]
        if not orphans:
            return 0
        return await self._vectors.delete(orphans)

    @staticmethod
    def _embed_text(norm: str, definition: str) -> str:
        """What gets embedded for one sense: the display headword and its definition."""
        return f"{render(norm)}: {definition}"

    # --- relation resolution ------------------------------------------------

    async def resolve_relations(self, batch_size: int = 20) -> list[BatchResult]:
        """Reconcile one batch of pending sense-relation edges.

        This is the manual and backfill entry point, for words that were done before
        the hook existed or whose hook was skipped.
        """
        return await self.resolve(batch_size, word_ids=None)

    async def resolve_inbound(self, word_ids: Sequence[int]) -> list[BatchResult]:
        """Resolve edges pointing at words that just became done.

        Every error is swallowed: the generation that triggered this is already
        committed, so a judge outage must degrade to leaving the edges pending.
        Coverage then grows with traffic and needs no scheduler.
        """
        if not word_ids:
            return []
        try:
            return await self.resolve(len(word_ids) * _INBOUND_FACTOR, list(word_ids))
        except Exception:  # noqa: BLE001 - inbound resolve is strictly best-effort
            return []

    async def resolve(self, batch_size: int, word_ids: Sequence[int] | None) -> list[BatchResult]:
        """Read the pending queue, judge it in one call, apply the verdicts.

        With no judge configured this returns nothing rather than failing, matching
        how the other model-backed surfaces degrade.
        """
        judge = self._judge_factory()
        if judge is None:
            return []
        from lexi_ai.generation.schemas import WsdCandidate, WsdTask
        from lexi_ai.generation.wsd import WSD_BATCH_CEIL, pos_filtered_candidates

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

        An out-of-range or absent index means unresolvable, never a guess: writing a
        resolution the model did not choose would point the edge at the wrong sense.
        """
        if chosen is None or not (0 <= chosen < len(candidates)):
            return ResolveDecision(task.edge_id, None, None)
        target = candidates[chosen]
        return ResolveDecision(task.edge_id, target.sense_id, sense_content_hash(target.definition))
