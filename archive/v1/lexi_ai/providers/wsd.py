"""WSD judge — the LLM half of sense-relation reconciliation.

Turns a list of :class:`WsdTask` (a source gloss + its POS-filtered target-sense
candidates) into a list of :class:`WsdChoice`. The model is injectable so tests pass a
fake and never touch the network. The judge NEVER trusts the returned index:
order-alignment (``choices[i]`` ↔ ``tasks[i]``) is repaired here (pad/truncate to the
task count); bounds-checking of ``chosen_index`` happens at the apply site.
"""

from collections.abc import Iterable, Sequence

from lexi_ai.prompts import PromptLoader
from lexi_ai.providers.seam import StructuredLLM, ainvoke_structured, guarded_messages
from lexi_ai.schemas import WsdBatch, WsdChoice, WsdTask
from lexi_ai.vocab import WSD_BATCH_CEIL, normalize_pos

# WSD_BATCH_CEIL (from ``lexi_ai.vocab``) is re-exported for callers importing it here.
__all__ = ["WsdJudge", "pos_filtered_candidates", "WSD_BATCH_CEIL"]


def pos_filtered_candidates(source_pos: str | None, candidates: Sequence):
    """Select which target-sense candidates to show the judge by POS.

    ``normalize_pos`` is applied to BOTH sides so ``adj`` vs ``adjective`` never
    mis-filters. Source POS unknown/unmappable → show ALL candidates (can't filter
    safely). Otherwise, if ≥1 candidate has a CLEAR same-POS match → keep the same-POS
    ones PLUS any unknown-POS candidates (NULL/legacy POS is never hard-excluded),
    dropping only clearly-different-POS candidates. If NO candidate has a clear same-POS
    match → show ALL: cross-POS relations like ``see_also`` exist, so never mass-drop to
    ``unresolvable`` on POS alone.
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

        One LLM call per batch. The returned list is normalized to exactly ``len(tasks)``:
        a short list is padded with ``chosen_index=None`` (treated as unresolvable), a long
        list is truncated.

        Scope note — COUNT only, not order: ``_align`` guarantees the returned list LENGTH
        matches ``tasks``; it does NOT detect or repair a REORDER. Alignment is purely
        positional — choice[i] is assumed to answer task[i]. The prompt numbers each task,
        shows candidate indices, and says "in order", and the chosen index is
        server-validated on apply, but a model that silently permutes its answers would
        mis-map. No reorder has been observed; if one ever is, add a ``task_index`` echo to
        :class:`WsdChoice` and re-key on apply."""
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
