"""Finding words: reference-anchored lookup and meaning-based ranking.

Both surfaces are free — neither generates an entry, and semantic search only
encodes the short query.

The two hit kinds in one ranked list are deliberate. A reference word that has
already been generated is folded into its generated hit rather than offered
again, so a caller cannot accidentally regenerate an entry it already has.
"""

from collections.abc import Callable

from lexi_ai.domain.errors import SemanticSearchDisabled
from lexi_ai.domain.models import SemanticSenseRow
from lexi_ai.domain.ports import UnitOfWork, VectorIndex
from lexi_ai.normalize import render
from lexi_ai.read_models import SearchResult, SemanticHit, SenseView

# The index is asked for more than k so that vectors whose sense has since been
# deleted or regenerated cannot squeeze real hits out of the answer. They are
# dropped during hydration, and a backfill prunes them for good.
_OVERFETCH = 10


class SearchService:
    """Lookup use cases over the reference loader and the unit of work."""

    def __init__(
        self,
        uow_factory: Callable[[], UnitOfWork],
        loader,  # noqa: ANN001 - the reference loader (Cambridge + WordNet)
        embedder,  # noqa: ANN001 - the encoder; only model_name and embed_one are used
        vectors: VectorIndex | None,
    ) -> None:
        self._uow = uow_factory
        self._loader = loader
        self._embedder = embedder
        self._vectors = vectors

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

    async def _reference_candidates(self, query: str) -> list[tuple[int, str, str | None, float]]:
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

        The vector index ranks; the relational store hydrates what it returned.
        Only vectors from the CURRENT encoder model are considered, so switching
        models yields nothing until a backfill runs rather than ranking against a
        different geometry.

        RAISES on encoder or index failure — a missing extra, a model that will not
        load, a device error, an unreachable index — and ``SemanticSearchDisabled``
        when the feature is off. An empty list means exactly one thing: nothing
        matched. Swallowing the failure instead would report "no results" for a
        broken or unconfigured installation, which reads as "this word is not in
        the dictionary" and is indistinguishable from the truthful answer.
        """
        if self._vectors is None:
            raise SemanticSearchDisabled(
                "semantic_search() is an opt-in feature and is off. Enable it with "
                "LEXI_VECTOR_BACKEND=lancedb (durable, needs the '[lancedb]' extra) "
                "or 'memory' (non-durable, tests only), plus the '[embeddings]' "
                "extra for the encoder"
            )
        if k <= 0:
            return []
        query_vector = await self._embedder.embed_one(query)
        hits = await self._vectors.query(
            query_vector, k + _OVERFETCH, {"model": self._embedder.model_name}
        )
        scores = {hit.id: hit.score for hit in hits}
        sense_ids = [int(hit.id) for hit in hits if hit.id.isdigit()]
        if not sense_ids:
            return []
        async with self._uow() as uow:
            rows = await uow.senses.semantic_rows(sense_ids)
        return [self._semantic_hit(scores[str(row.sense_id)], row) for row in rows[:k]]

    @staticmethod
    def _semantic_hit(score: float, row: SemanticSenseRow) -> SemanticHit:
        return SemanticHit(
            lexi_word_id=row.word_id,
            display=render(row.norm),
            entry_type=row.entry_type,
            score=score,
            sense=SenseView(definition=row.definition, tier=row.tier, pos=None, cefr_level=None),
        )
