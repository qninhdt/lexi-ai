"""Distractor provider — wrong-option source for MCQ question plugins.

A two-step best-effort ladder, reusing existing infrastructure (DRY):

1. **Semantic neighbours** — the vector index ranks other done senses against the
   target's core-sense vector. Only fires when that sense is embedded; empty
   otherwise, never an error — the same posture as semantic search.
2. **Topic fallback** — words sharing one of the entry's topic tags (reuses the
   tag repository's ``words_for_key`` read).

Options are deduped by ``match_key`` and exclude the target word and its aliases,
so the correct answer (or a surface variant of it) can never slip in as a
distractor. The caller degrades the option count when the ladder returns fewer
than requested.
"""

from lexi_ai.normalize import render, tag_key
from lexi_ai.questions.dedup import DistractorDedup
from lexi_ai.read_models import Entry

# Cap the per-tag fetch so a huge topic doesn't dominate the candidate pool.
_TAG_FETCH_LIMIT = 50
# How many neighbours to ask the index for. Every sense of the target word is a
# near-certain top hit and gets discarded, so the pool needs headroom over the
# option count the caller will consume.
_SEMANTIC_FETCH_LIMIT = 50


class DistractorProvider:
    """Best-effort wrong-option source, shared by every MCQ plugin."""

    def __init__(self, uow_factory, embedder, vectors):
        # ``uow_factory``: callable returning a unit of work (read-only here).
        # ``embedder``: lexi_ai.embeddings.Embedder (only its model_name is used).
        # ``vectors``: the VectorIndex holding sense vectors.
        self._uow_factory = uow_factory
        self._embedder = embedder
        self._vectors = vectors

    async def for_word(self, entry: Entry, *, k: int, pos: str | None = None) -> list[str]:
        """Up to ``k`` distinct distractor display strings for ``entry``'s core sense.

        ``pos`` is accepted for interface stability and future POS-aware ranking;
        the topic fallback can't filter on it (the tag query returns entry_type,
        not part-of-speech), so v1 leans on semantic + topic relatedness. Never
        raises: any source failure degrades to fewer options.
        """
        if k <= 0:
            return []
        # The target word + aliases can never be a distractor, and each display is
        # deduped by match_key.
        dedup = DistractorDedup(entry)
        for display in await self._semantic(entry):
            if dedup.take(display) and len(dedup.items) >= k:
                return dedup.items
        for display in await self._by_topics(entry):
            if dedup.take(display) and len(dedup.items) >= k:
                return dedup.items
        return dedup.items

    # --- ladder steps (each best-effort, returns [] on any miss/failure) ------

    async def _semantic(self, entry: Entry) -> list[str]:
        """Displays of the nearest senses belonging to OTHER words, best first.

        Skipped entirely when semantic search is off, which is a configuration
        fact rather than a failure — the ladder falls through to topic tags. Being
        explicit here keeps the rung from depending on an ``AttributeError`` landing
        in the catch-all below.
        """
        if self._vectors is None:
            return []
        try:
            target = await self._target_vector(entry)
            if target is None:
                return []
            hits = await self._vectors.query(
                target, _SEMANTIC_FETCH_LIMIT, {"model": self._embedder.model_name}
            )
            sense_ids = [int(hit.id) for hit in hits if hit.id.isdigit()]
            if not sense_ids:
                return []
            async with self._uow_factory() as uow:
                rows = await uow.senses.semantic_rows(sense_ids)
        except Exception:  # noqa: BLE001 - best-effort, like semantic search
            return []
        return [render(row.norm) for row in rows if row.word_id != entry.word_id]

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

    # --- helpers --------------------------------------------------------------

    async def _target_vector(self, entry: Entry) -> list[float] | None:
        """The target's core-sense vector, or any embedded sense of it.

        Senses are core-first, so the first one that is indexed is the best anchor
        available. ``None`` when the word has no vector at all — the ladder then
        falls through to topics rather than ranking against something arbitrary.
        """
        sense_ids = [sense.sense_id for sense in entry.senses if sense.sense_id is not None]
        if not sense_ids:
            return None
        stored = await self._vectors.fetch([str(sense_id) for sense_id in sense_ids])
        for sense_id in sense_ids:
            vector = stored.get(str(sense_id))
            if vector is not None:
                return vector
        return None
