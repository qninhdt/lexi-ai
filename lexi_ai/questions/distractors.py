"""Distractor provider — wrong-option source for MCQ question plugins.

A two-step best-effort ladder, reusing existing infrastructure (DRY):

1. **Semantic neighbours** — rank OTHER done senses by cosine similarity to the
   target's core-sense vector (reuses ``Repository.embedded_senses`` +
   ``vectors.cosine``). Only fires when embeddings exist; empty otherwise, never
   an error — same posture as :meth:`Lexicon.semantic_search`.
2. **Topic fallback** — words sharing one of the entry's topic tags (reuses
   ``Repository.words_for_tag_key``).

Options are deduped by ``match_key`` and exclude the target word and its aliases,
so the correct answer (or a surface variant of it) can never slip in as a
distractor. The caller degrades the option count when the ladder returns fewer
than requested.
"""

from lexi_ai.normalize import match_key, render, tag_key
from lexi_ai.read_models import Entry
from lexi_ai.vectors import cosine, unpack_vector

# Cap the per-tag fetch so a huge topic doesn't dominate the candidate pool.
_TAG_FETCH_LIMIT = 50


class DistractorProvider:
    """Best-effort wrong-option source, shared by every MCQ plugin."""

    def __init__(self, repo, embedder):
        # ``repo``: lexi_ai.persistence.repository.Repository (read-only here).
        # ``embedder``: lexi_ai.embeddings.Embedder (only its model_name is used).
        self._repo = repo
        self._embedder = embedder

    async def for_word(self, entry: Entry, *, k: int, pos: str | None = None) -> list[str]:
        """Up to ``k`` distinct distractor display strings for ``entry``'s core sense.

        ``pos`` is accepted for interface stability and future POS-aware ranking;
        the topic fallback can't filter on it (the tag query returns entry_type,
        not part-of-speech), so v1 leans on semantic + topic relatedness. Never
        raises: any source failure degrades to fewer options.
        """
        if k <= 0:
            return []
        seen = self._exclude_keys(entry)
        out: list[str] = []
        for display in await self._semantic(entry):
            if self._take(display, seen, out) and len(out) >= k:
                return out
        for display in await self._by_topics(entry):
            if self._take(display, seen, out) and len(out) >= k:
                return out
        return out

    # --- ladder steps (each best-effort, returns [] on any miss/failure) ------

    async def _semantic(self, entry: Entry) -> list[str]:
        """Other-word sense displays, ranked by cosine to the target core sense."""
        try:
            rows = await self._repo.embedded_senses(self._embedder.model_name)
        except Exception:  # noqa: BLE001 - best-effort, like semantic_search
            return []
        if not rows:
            return []
        target = self._target_vector(entry, rows)
        if target is None:
            return []
        scored = sorted(
            (
                (cosine(target, unpack_vector(r.embedding)), r)
                for r in rows
                if r.word_id != entry.word_id
            ),
            key=lambda s: -s[0],
        )
        return [render(r.norm) for _score, r in scored]

    async def _by_topics(self, entry: Entry) -> list[str]:
        """Displays of words sharing one of the entry's topic tags."""
        out: list[str] = []
        for topic in entry.topics:
            try:
                rows = await self._repo.words_for_tag_key(
                    tag_key(topic.name), limit=_TAG_FETCH_LIMIT
                )
            except Exception:  # noqa: BLE001 - best-effort
                continue
            out.extend(render(norm) for _wid, norm, _etype in rows)
        return out

    # --- helpers --------------------------------------------------------------

    @staticmethod
    def _target_vector(entry: Entry, rows) -> list[float] | None:
        """The target word's core-sense vector, if it is embedded.

        Prefers the row matching the entry's core sense id (senses are core-first);
        falls back to any embedded sense of the target word.
        """
        own = [r for r in rows if r.word_id == entry.word_id]
        if not own:
            return None
        core_id = entry.senses[0].sense_id if entry.senses else None
        for r in own:
            if core_id is not None and r.sense_id == core_id:
                return unpack_vector(r.embedding)
        return unpack_vector(own[0].embedding)

    @staticmethod
    def _exclude_keys(entry: Entry) -> set[str]:
        """match_keys that must never appear as a distractor: the word + aliases."""
        keys = {match_key(entry.norm)}
        keys.update(match_key(a.alias_norm) for a in entry.aliases)
        return keys

    @staticmethod
    def _take(display: str, seen: set[str], out: list[str]) -> bool:
        """Append ``display`` if its match_key is new; return whether it was taken."""
        key = match_key(display)
        if not key or key in seen:
            return False
        seen.add(key)
        out.append(display)
        return True
