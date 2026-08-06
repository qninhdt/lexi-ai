"""A bounded embedding backfill must not read the whole candidate table.

``embed_missing(limit=N)`` used to ask the repository for every done sense, filter
out the ones the vector index already held, and slice the result to N. Correct, and
proportional to the dictionary rather than to N: a `limit=32` request against
200,000 senses loaded 200,000 rows to keep 32.

The fix pages through candidates with a cursor. The subtlety worth testing is why a
plain ``LIMIT N`` is wrong: the already-embedded set lives in the vector index, not
the database, so the first N candidates can all be embedded already. That page
looks empty after filtering, which a naive implementation reads as "nothing left to
do" while unembedded senses sit further down the table.
"""

from dataclasses import dataclass

import pytest

from lexi_ai.application.enrichment import EnrichmentService
from lexi_ai.domain.models import SenseEmbeddingNeed

pytestmark = pytest.mark.asyncio


@dataclass
class _Call:
    limit: int | None
    after_sense_id: int | None


class RecordingSenseRepo:
    """A candidate table that honours limit + cursor and records every request."""

    def __init__(self, sense_ids: list[int]) -> None:
        self._ids = sorted(sense_ids)
        self.calls: list[_Call] = []

    async def needing_embedding(
        self,
        word_ids: list[int] | None = None,
        limit: int | None = None,
        after_sense_id: int | None = None,
    ) -> list[SenseEmbeddingNeed]:
        self.calls.append(_Call(limit, after_sense_id))
        rows = self._ids
        if after_sense_id is not None:
            rows = [sid for sid in rows if sid > after_sense_id]
        if limit is not None:
            rows = rows[:limit]
        return [SenseEmbeddingNeed(sid, f"w{sid}", "a definition") for sid in rows]

    @property
    def rows_read(self) -> int:
        """How many rows the repository was asked to materialise in total."""
        total = 0
        for call in self.calls:
            available = self._ids
            if call.after_sense_id is not None:
                available = [s for s in available if s > call.after_sense_id]
            total += len(available) if call.limit is None else min(call.limit, len(available))
        return total


class _Uow:
    def __init__(self, senses: RecordingSenseRepo) -> None:
        self.senses = senses

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _Embedder:
    model_name = "test-model"

    def __init__(self) -> None:
        self.embedded: list[list[str]] = []

    async def embed(self, texts):
        self.embedded.append(list(texts))
        return [[0.1, 0.2] for _ in texts]


class _Index:
    def __init__(self, stored: set[str]) -> None:
        self._stored = stored
        self.upserted: list[list] = []

    async def ids(self, _where=None) -> set[str]:
        return set(self._stored)

    async def upsert(self, records) -> int:
        batch = list(records)
        self.upserted.append(batch)
        return len(batch)


def _service(repo: RecordingSenseRepo, index: _Index, embedder: _Embedder):
    def _unused(*_a, **_k):  # pragma: no cover - not reached by these paths
        raise AssertionError("unexpected call")

    return EnrichmentService(
        lambda: _Uow(repo),
        embedder,
        lambda: None,
        lambda: None,
        _unused,
        _unused,
        12,
        index,
    )


async def test_a_bounded_backfill_does_not_read_the_whole_table():
    """The row count read must scale with the limit, not with the dictionary."""
    repo = RecordingSenseRepo(list(range(1, 20_001)))
    index = _Index(stored=set())
    embedder = _Embedder()

    written = await _service(repo, index, embedder).embed_missing(limit=8)

    assert written == 8
    assert [record.id for record in index.upserted[0]] == [str(i) for i in range(1, 9)]
    # The defect read all 20,000. A page is oversized against the shortfall, so
    # allow generous headroom while still ruling out a full-table read.
    assert repo.rows_read < 1_000, f"read {repo.rows_read} rows for a limit of 8"
    assert all(call.limit is not None for call in repo.calls), "a page was unbounded"


async def test_pages_past_senses_the_index_already_holds():
    """A fully-embedded first page must not read as 'nothing left to do'.

    This is what a plain `LIMIT N` gets wrong. Senses 1-500 are already embedded,
    so the first page filters down to nothing; the unembedded rows begin at 501.
    """
    repo = RecordingSenseRepo(list(range(1, 1_001)))
    index = _Index(stored={str(i) for i in range(1, 501)})
    embedder = _Embedder()

    written = await _service(repo, index, embedder).embed_missing(limit=5)

    assert written == 5
    assert [record.id for record in index.upserted[0]] == ["501", "502", "503", "504", "505"]
    # More than one request: the first page was consumed entirely by the filter.
    assert len(repo.calls) > 1, "gave up after a single fully-embedded page"
    assert repo.calls[1].after_sense_id is not None, "second page did not use a cursor"


async def test_stops_when_the_table_runs_out_rather_than_looping():
    """Fewer unembedded rows than requested must terminate, not spin."""
    repo = RecordingSenseRepo([1, 2, 3])
    index = _Index(stored={"1"})
    embedder = _Embedder()

    written = await _service(repo, index, embedder).embed_missing(limit=50)

    assert written == 2
    assert [record.id for record in index.upserted[0]] == ["2", "3"]


async def test_an_unlimited_backfill_still_reads_every_candidate():
    """limit=None is a whole-dictionary backfill and must stay exhaustive."""
    repo = RecordingSenseRepo(list(range(1, 101)))
    index = _Index(stored=set())
    embedder = _Embedder()

    written = await _service(repo, index, embedder).embed_missing()

    assert written == 100
    assert repo.calls == [_Call(limit=None, after_sense_id=None)]


async def test_returns_zero_without_touching_the_encoder_when_nothing_is_pending():
    repo = RecordingSenseRepo([1, 2])
    index = _Index(stored={"1", "2"})
    embedder = _Embedder()

    assert await _service(repo, index, embedder).embed_missing(limit=10) == 0
    assert embedder.embedded == [], "the encoder ran for an empty batch"
    assert index.upserted == []
