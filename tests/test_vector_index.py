"""One contract, exercised against every vector backend.

The suite is parametrized over the adapters rather than written per adapter: the
whole point of the port is that swapping backends changes nothing upstream, and a
test that only covers the in-memory fake would not prove that. The exact-scan
in-memory index doubles as the ground truth the approximate backend is compared
against.
"""

import importlib.util

import pytest

from lexi_ai.config import Settings
from lexi_ai.domain.models import VectorRecord
from lexi_ai.infrastructure.vectors import build_vector_index
from lexi_ai.infrastructure.vectors.memory_index import InMemoryVectorIndex

_HAS_LANCEDB = importlib.util.find_spec("lancedb") is not None

lancedb_only = pytest.mark.skipif(
    not _HAS_LANCEDB, reason="needs the lancedb extra (uv sync --extra lancedb)"
)


@pytest.fixture(params=["memory", "lancedb"])
def index(request, tmp_path):
    """One index per backend, each isolated to this test."""
    if request.param == "memory":
        return InMemoryVectorIndex()
    if not _HAS_LANCEDB:
        pytest.skip("needs the lancedb extra (uv sync --extra lancedb)")
    from lexi_ai.infrastructure.vectors.lancedb_index import LanceDbVectorIndex

    return LanceDbVectorIndex(str(tmp_path / "vectors"))


def _record(id_: str, vector: list[float], model: str = "m1") -> VectorRecord:
    return VectorRecord(id=id_, vector=vector, meta={"model": model})


async def test_query_ranks_by_similarity_best_first(index):
    await index.upsert(
        [
            _record("1", [1.0, 0.0, 0.0]),
            _record("2", [0.8, 0.6, 0.0]),
            _record("3", [0.0, 0.0, 1.0]),
        ]
    )

    hits = await index.query([1.0, 0.0, 0.0], 3)

    assert [hit.id for hit in hits] == ["1", "2", "3"]
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


async def test_query_honours_k_and_rejects_nothing_below_one(index):
    await index.upsert([_record("1", [1.0, 0.0]), _record("2", [0.0, 1.0])])

    assert len(await index.query([1.0, 0.0], 1)) == 1
    assert await index.query([1.0, 0.0], 0) == []
    assert await index.query([1.0, 0.0], -5) == []


async def test_query_prefilters_on_metadata(index):
    """A wrong-model vector must not consume one of the k slots."""
    await index.upsert([_record("1", [1.0, 0.0], model="m1"), _record("2", [1.0, 0.0], model="m2")])

    hits = await index.query([1.0, 0.0], 1, {"model": "m2"})

    assert [hit.id for hit in hits] == ["2"]


async def test_upsert_replaces_a_vector_in_place(index):
    await index.upsert([_record("1", [1.0, 0.0], model="m1")])
    await index.upsert([_record("1", [0.0, 1.0], model="m2")])

    assert await index.ids() == {"1"}
    assert await index.ids({"model": "m1"}) == set()
    assert (await index.fetch(["1"]))["1"] == pytest.approx([0.0, 1.0])


async def test_ids_filters_and_delete_removes(index):
    await index.upsert(
        [
            _record("1", [1.0, 0.0], model="m1"),
            _record("2", [0.0, 1.0], model="m1"),
            _record("3", [1.0, 1.0], model="m2"),
        ]
    )

    assert await index.ids() == {"1", "2", "3"}
    assert await index.ids({"model": "m1"}) == {"1", "2"}
    assert await index.delete(["2", "3"]) == 2
    assert await index.ids() == {"1"}


async def test_fetch_omits_ids_the_index_does_not_hold(index):
    await index.upsert([_record("7", [1.0, 0.0])])

    fetched = await index.fetch(["7", "9999"])

    assert set(fetched) == {"7"}
    assert fetched["7"] == pytest.approx([1.0, 0.0])


async def test_reads_on_an_empty_index_are_empty_not_errors(index):
    """Semantic search must degrade, so nothing here may raise before a first write."""
    assert await index.query([1.0, 0.0], 5) == []
    assert await index.ids() == set()
    assert await index.fetch(["1"]) == {}
    assert await index.delete(["1"]) == 0


async def test_empty_writes_are_no_ops(index):
    assert await index.upsert([]) == 0
    assert await index.delete([]) == 0


async def test_a_mixed_dimension_upsert_is_rejected(index):
    with pytest.raises(ValueError, match="dimension"):
        await index.upsert([_record("1", [1.0, 0.0]), _record("2", [1.0, 0.0, 0.0])])


# --- backend selection ------------------------------------------------------


def test_backend_selection_is_settings_only():
    assert isinstance(build_vector_index(Settings(vector_backend="memory")), InMemoryVectorIndex)


@lancedb_only
def test_lancedb_is_the_declared_default_backend(tmp_path):
    """The durable backend is the default; the test tier opts out via env."""
    from lexi_ai.infrastructure.vectors.lancedb_index import LanceDbVectorIndex

    assert Settings.model_fields["vector_backend"].default == "lancedb"
    built = build_vector_index(Settings(vector_backend="lancedb", vector_path=str(tmp_path)))

    assert isinstance(built, LanceDbVectorIndex)


def test_an_unknown_backend_fails_loudly():
    with pytest.raises(ValueError, match="unknown vector backend"):
        build_vector_index(Settings(vector_backend="pinecone"))


def test_constructing_the_lancedb_adapter_imports_nothing_heavy(tmp_path):
    """The adapter must be cheap to build so a base install can wire it and degrade.

    Nothing touches the filesystem or imports lancedb until a vector is actually
    used, which is what lets an install without the extra construct the graph.
    """
    from lexi_ai.infrastructure.vectors.lancedb_index import LanceDbVectorIndex

    store = tmp_path / "not-created-yet"
    LanceDbVectorIndex(str(store))

    assert not store.exists()


async def test_the_memory_index_starts_empty_per_instance():
    """Non-durability is the contract, not an accident: a fresh process re-backfills."""
    first = InMemoryVectorIndex()
    await first.upsert([_record("1", [1.0])])

    assert await InMemoryVectorIndex().ids() == set()
    assert await first.ids() == {"1"}
