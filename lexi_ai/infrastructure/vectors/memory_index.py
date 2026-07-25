"""An in-process vector index: the hermetic default for tests and small runs.

Ranking is an exact scan with the pure-Python cosine, so results are the ground
truth an approximate backend is checked against. Nothing is persisted — a new
process starts empty and the backfill refills it, which is exactly the
eventual-consistency posture the port promises.
"""

from collections.abc import Mapping, Sequence

from lexi_ai.domain.models import VectorHit, VectorRecord
from lexi_ai.infrastructure.vectors.validation import uniform_dimension
from lexi_ai.vectors import cosine


class InMemoryVectorIndex:
    """Exact-scan vector index held in a dict. Not durable, not concurrent-safe."""

    def __init__(self) -> None:
        self._vectors: dict[str, list[float]] = {}
        self._meta: dict[str, dict[str, str]] = {}

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        if not records:
            return 0
        uniform_dimension(records)
        for record in records:
            self._vectors[record.id] = list(record.vector)
            self._meta[record.id] = dict(record.meta)
        return len(records)

    async def query(
        self, vector: Sequence[float], k: int, where: Mapping[str, str] | None = None
    ) -> list[VectorHit]:
        if k <= 0:
            return []
        query_vector = list(vector)
        scored = [
            VectorHit(id=stored_id, score=cosine(query_vector, stored))
            for stored_id, stored in self._vectors.items()
            if self._matches(stored_id, where)
        ]
        scored.sort(key=lambda hit: -hit.score)
        return scored[:k]

    async def delete(self, ids: Sequence[str]) -> int:
        removed = 0
        for stored_id in ids:
            if self._vectors.pop(stored_id, None) is not None:
                self._meta.pop(stored_id, None)
                removed += 1
        return removed

    async def ids(self, where: Mapping[str, str] | None = None) -> set[str]:
        return {stored_id for stored_id in self._vectors if self._matches(stored_id, where)}

    async def fetch(self, ids: Sequence[str]) -> dict[str, list[float]]:
        return {
            stored_id: list(self._vectors[stored_id])
            for stored_id in ids
            if stored_id in self._vectors
        }

    def _matches(self, stored_id: str, where: Mapping[str, str] | None) -> bool:
        if not where:
            return True
        meta = self._meta.get(stored_id, {})
        return all(meta.get(key) == value for key, value in where.items())
