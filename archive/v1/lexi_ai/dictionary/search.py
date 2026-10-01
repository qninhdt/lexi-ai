"""Finding words: reference-anchored lookup. Never generates an entry.

The two hit kinds in one ranked list are deliberate: a reference word that has already been
generated is folded into its generated hit rather than offered again, so a caller cannot
accidentally regenerate an entry it already has.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from lexi_ai.models import SearchResult
from lexi_ai.text import render

if TYPE_CHECKING:
    from lexi_ai.db.session import SqlAlchemyUnitOfWork


class SearchService:
    """Lookup use cases over the reference loader and the unit of work."""

    def __init__(
        self,
        uow_factory: Callable[[], SqlAlchemyUnitOfWork],
        loader,  # noqa: ANN001 - the reference loader (Cambridge + WordNet)
    ) -> None:
        self._uow = uow_factory
        self._loader = loader

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
        """Reference matches for a query: exact first at full score, then fuzzy, deduped by
        reference id keeping the first-seen (best) score.
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
