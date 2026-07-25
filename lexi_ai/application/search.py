"""Finding words: reference-anchored lookup and meaning-based ranking.

Both surfaces are free — neither generates an entry, and semantic search only
encodes the short query.

The two hit kinds in one ranked list are deliberate. A reference word that has
already been generated is folded into its generated hit rather than offered
again, so a caller cannot accidentally regenerate an entry it already has.
"""

from collections.abc import Callable

from lexi_ai.domain.ports import UnitOfWork
from lexi_ai.normalize import render
from lexi_ai.read_models import SearchResult, SemanticHit, SenseView
from lexi_ai.vectors import cosine, unpack_vector


class SearchService:
    """Lookup use cases over the reference loader and the unit of work."""

    def __init__(
        self,
        uow_factory: Callable[[], UnitOfWork],
        loader,  # noqa: ANN001 - the reference loader (Cambridge + WordNet)
        embedder,  # noqa: ANN001 - the encoder; only model_name and embed_one are used
    ) -> None:
        self._uow = uow_factory
        self._loader = loader
        self._embedder = embedder

    async def search(self, query: str) -> list[SearchResult]:
        """One ranked list mixing generated entries and generatable suggestions."""
        candidates = await self._reference_candidates(query)
        cambridge_ids = [candidate[0] for candidate in candidates]
        async with self._uow() as uow:
            generated = await uow.words.generated_by_cambridge(cambridge_ids)
        glosses = await self._loader.cambridge.first_definitions(cambridge_ids)

        results: list[SearchResult] = []
        seen_words: set[int] = set()
        for cambridge_id, display, entry_type, score in candidates:
            hit = generated.get(cambridge_id)
            if hit is not None:
                if hit.word_id in seen_words:
                    continue  # two reference ids fold onto one generated word
                seen_words.add(hit.word_id)
                results.append(
                    SearchResult(
                        # Display is always rendered from the lemma; the stored norm
                        # keeps placeholders like {sb} that a caller must not see.
                        display=render(hit.norm),
                        entry_type=hit.entry_type,
                        score=score,
                        lexi_word_id=hit.word_id,
                    )
                )
            else:
                results.append(
                    SearchResult(
                        display=display,
                        entry_type=entry_type,
                        score=score,
                        cambridge_id=cambridge_id,
                        gloss=glosses.get(cambridge_id),
                    )
                )
        results.sort(key=lambda result: (-result.score, result.display))
        return results

    async def _reference_candidates(
        self, query: str
    ) -> list[tuple[int, str, str | None, float]]:
        """Reference matches for a query: exact first at full score, then fuzzy.

        Deduped by reference id keeping the best score, which is the first seen
        given the ordering.
        """
        exact = await self._loader.cambridge.resolve_exact(query)
        exact_ids = {reference.word_id for reference in exact}
        ranked = await self._loader.cambridge.rank_similar(query)
        candidates = [
            (reference.word_id, reference.display_form, reference.entry_type, 1.0)
            for reference in exact
        ]
        candidates += [
            (reference.word_id, reference.display_form, reference.entry_type, score)
            for reference, score in ranked
            if reference.word_id not in exact_ids
        ]
        seen: set[int] = set()
        deduped: list[tuple[int, str, str | None, float]] = []
        for candidate in candidates:
            if candidate[0] not in seen:
                seen.add(candidate[0])
                deduped.append(candidate)
        return deduped

    async def semantic_search(self, query: str, k: int = 10) -> list[SemanticHit]:
        """Rank generated senses by meaning similarity to a query.

        Only senses carrying a vector from the CURRENT model are considered, so
        switching models degrades this to empty until a backfill runs rather than
        ranking against stale vectors. An encoder failure degrades to empty too.
        """
        if k <= 0:
            return []
        async with self._uow() as uow:
            rows = await uow.senses.embedded(self._embedder.model_name)
        if not rows:
            return []
        try:
            query_vector = await self._embedder.embed_one(query)
        except Exception:  # noqa: BLE001 - best-effort: degrade to [] on encode failure
            return []
        scored = sorted(
            ((cosine(query_vector, unpack_vector(row.embedding)), row) for row in rows),
            key=lambda pair: -pair[0],
        )
        return [self._semantic_hit(score, row) for score, row in scored[:k]]

    @staticmethod
    def _semantic_hit(score: float, row) -> SemanticHit:  # noqa: ANN001 - an EmbeddedSenseRow
        return SemanticHit(
            lexi_word_id=row.word_id,
            display=render(row.norm),
            entry_type=row.entry_type,
            score=score,
            sense=SenseView(
                definition=row.definition, tier=row.tier, pos=None, cefr_level=None
            ),
        )
