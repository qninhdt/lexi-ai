"""WSD judge — the LLM half of sense-relation reconciliation (Phase 4).

A thin wrapper over the injectable :class:`StructuredLLM` seam that turns a list
of :class:`WsdTask` (a source gloss + its POS-filtered target-sense candidates)
into a list of :class:`WsdChoice` (the chosen candidate index, or ``None``). The
model is injectable so tests pass a fake and never touch the network, mirroring
:class:`~lexi_ai.generation.generator.Generator`.

The judge NEVER trusts the returned index: order-alignment (``choices[i]`` ↔
``tasks[i]``) is repaired here (pad/truncate to the task count); bounds-checking
of ``chosen_index`` happens at the apply site ([F3]).
"""

from collections.abc import Iterable, Sequence

from lexi_ai.constants import normalize_pos
from lexi_ai.generation.schemas import WsdBatch, WsdChoice, WsdTask
from lexi_ai.llm import StructuredLLM, ainvoke_structured, guarded_messages
from lexi_ai.prompts import PromptLoader

# [F9] cost guards — both caller/data-controlled, so both are clamped: at most
# WSD_BATCH_CEIL edges per LLM call, at most WSD_CANDIDATE_CAP target senses per
# task (top-K by sense_order). A batch_size=1000 request is silently clamped.
WSD_BATCH_CEIL = 50
WSD_CANDIDATE_CAP = 12


def pos_filtered_candidates(source_pos: str | None, candidates: Sequence):
    """Select which target-sense candidates to show the judge, by POS ([F2]).

    ``normalize_pos`` is applied to BOTH sides so ``adj`` vs ``adjective`` never
    mis-filters. Rules (all deliberate, see phase-04 F2/F10):

    - Source POS unknown/unmappable → show ALL candidates (can't filter safely).
    - Otherwise, if ≥1 candidate has a CLEAR same-POS match → keep the same-POS
      ones PLUS any unknown-POS candidates (NULL/legacy POS is never hard-excluded),
      dropping only clearly-different-POS candidates.
    - If NO candidate has a clear same-POS match → show ALL (cross-POS relations
      like ``see_also`` exist; never mass-drop to ``unresolvable`` on POS alone).

    Never returns fewer than ``candidates`` unless a clear same-POS subset exists.
    """
    src = normalize_pos(source_pos)
    if src is None:
        return list(candidates)
    has_clear_same = any(normalize_pos(c.pos) == src for c in candidates)
    if not has_clear_same:
        return list(candidates)
    # Keep same-POS + unknown-POS; drop only clearly-different known POS.
    return [c for c in candidates if normalize_pos(c.pos) in (src, None)]


class WsdJudge:
    """Batch word-sense-disambiguation judge over a :class:`StructuredLLM`."""

    def __init__(
        self,
        structured_llm: StructuredLLM,
        max_retries: int = 3,
        base_delay: float = 0.5,
    ):
        self._llm = structured_llm
        self._max_retries = max_retries
        self._base_delay = base_delay

    async def judge(self, tasks: Sequence[WsdTask]) -> list[WsdChoice]:
        """Return one :class:`WsdChoice` per task, order-aligned with ``tasks``.

        One LLM call per batch. The returned list is normalized to exactly
        ``len(tasks)``: a short list is padded with ``chosen_index=None`` (treated
        as unresolvable), a long list is truncated — so a mis-counted model
        response can never mis-map choices onto the wrong edges.
        """
        tasks = list(tasks)
        if not tasks:
            return []
        system = PromptLoader.render("wsd_system")
        user = PromptLoader.render("wsd_user", tasks=tasks)
        messages = guarded_messages(system, user)
        batch: WsdBatch = await ainvoke_structured(
            self._llm,
            messages,
            WsdBatch,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
        )
        return _align(batch.choices, len(tasks))


def _align(choices: Iterable[WsdChoice], n: int) -> list[WsdChoice]:
    out = list(choices)[:n]
    if len(out) < n:
        out += [WsdChoice(chosen_index=None) for _ in range(n - len(out))]
    return out
