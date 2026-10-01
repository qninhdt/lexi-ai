"""Distractor provider — wrong-option source for MCQ question plugins.

Options come from the target word's own topic tags: words sharing a topic are same-domain and
different-meaning, which is what a wrong option has to be. They are deduped by ``match_key`` and
exclude the target word and its aliases, so the correct answer (or a surface variant of it) can
never slip in.
"""

from lexi_ai.models import Entry
from lexi_ai.questions.dedup import DistractorDedup
from lexi_ai.text import render, tag_key

# Cap the per-tag fetch so a huge topic doesn't dominate the candidate pool.
_TAG_FETCH_LIMIT = 50


class DistractorProvider:
    """Best-effort wrong-option source, shared by every MCQ plugin."""

    def __init__(self, uow_factory):
        # Callable returning a unit of work (read-only here).
        self._uow_factory = uow_factory

    async def for_word(self, entry: Entry, *, k: int) -> list[str]:
        """Up to ``k`` distinct distractor display strings for ``entry``.

        This never raises: a source failure degrades to fewer options.
        """
        if k <= 0:
            return []
        dedup = DistractorDedup(entry)
        for display in await self._by_topics(entry):
            if dedup.take(display) and len(dedup.items) >= k:
                break
        return dedup.items

    async def _by_topics(self, entry: Entry) -> list[str]:
        """Displays of words sharing one of the entry's topic tags."""
        out: list[str] = []
        for topic in entry.topics:
            try:
                async with self._uow_factory() as uow:
                    rows = await uow.tags.words_for_key(tag_key(topic.name), limit=_TAG_FETCH_LIMIT)
            except Exception:  # noqa: BLE001 - best-effort
                continue
            out.extend(render(norm) for _wid, norm, _etype in rows)
        return out
