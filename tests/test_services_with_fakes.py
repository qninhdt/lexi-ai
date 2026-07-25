"""Every application service, constructed from Ports alone — no database.

This is the claim the service split was made for: a service depends on
`domain.ports`, not on SQLAlchemy. The rest of the suite exercises the services
through the facades against a real SQLite file, which proves they *work* but not
that they are *decoupled* — a service that quietly imported the ORM would pass
those tests unchanged.

So the unit of work here is a plain object. If any of these tests ever needs a
database to run, the decoupling has regressed.

The behaviours chosen are the orchestration branches that belong to the service
rather than to SQL: batch failure isolation, no-op guards, and degradation when a
collaborator is missing or broken. Anything whose correctness lives in a query is
tested against the real database instead, where a fake would only assert that the
fake was called.
"""

import pytest

from lexi_ai.application.dictionary import DictionaryService
from lexi_ai.application.enrichment import EnrichmentService
from lexi_ai.application.search import SearchService
from lexi_ai.application.tags import TagService
from lexi_ai.application.themes import ThemeService
from lexi_ai.domain.models import SemanticSenseRow, VectorHit
from lexi_ai.read_models import Entry, SenseView


class FakeRepo:
    """Answers any repository call from a dict of canned returns.

    Unknown calls raise rather than returning a mock, so a service reaching for a
    method this fake was not told about fails loudly instead of silently passing.
    """

    def __init__(self, **returns) -> None:
        self._returns = returns
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        async def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name not in self._returns:
                raise AssertionError(f"unexpected repository call: {name}")
            value = self._returns[name]
            return value(*args, **kwargs) if callable(value) else value

        return call


class FakeUnitOfWork:
    """A unit of work with no session, no engine, and no SQL."""

    def __init__(self, **repos) -> None:
        for aggregate in ("words", "senses", "themes", "tags", "entries", "stats"):
            setattr(self, aggregate, repos.get(aggregate, FakeRepo()))
        self.commits = 0
        self.rollbacks = 0

    def __call__(self) -> "FakeUnitOfWork":
        """The services take a factory; one instance is its own factory here so a
        test can inspect what happened across every `async with` block."""
        return self

    async def __aenter__(self) -> "FakeUnitOfWork":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1

    async def flush(self) -> None:
        return None


def _entry(word_id: int) -> Entry:
    return Entry(
        display=f"w{word_id}",
        norm=f"w{word_id}",
        entry_type="word",
        pos="noun",
        status="done",
        word_id=word_id,
    )


def _sense_row(sense_id: int, word_id: int, norm: str) -> SemanticSenseRow:
    return SemanticSenseRow(
        sense_id=sense_id,
        word_id=word_id,
        norm=norm,
        entry_type="word",
        definition=f"meaning of {norm}",
        tier="core",
    )


# --- dictionary -------------------------------------------------------------


async def test_batch_entry_reads_isolate_one_failure_from_its_siblings():
    """A bad id must be reported in its own slot, never abort the batch."""

    def entry(word_id, _overlay):
        if word_id == 2:
            raise ValueError("no such word")
        return _entry(word_id)

    uow = FakeUnitOfWork(entries=FakeRepo(entry=entry))
    service = DictionaryService(uow, _unused_resolver, _unused_index)

    results = await service.entries([1, 2, 3])

    assert [result.error is None for result in results] == [True, False, True]
    assert [result.value.word_id for result in results if result.value] == [1, 3]
    assert "no such word" in results[1].error


async def test_an_unknown_theme_raises_instead_of_falling_back_to_neutral():
    """Silently returning the neutral entry would hide the caller's mistake."""

    async def resolve(theme):
        raise ValueError(f"unknown theme: {theme!r}")

    service = DictionaryService(FakeUnitOfWork(), resolve, _unused_index)

    with pytest.raises(ValueError, match="unknown theme"):
        await service.entry(1, theme="nope")


async def test_reading_no_senses_asks_the_repository_nothing():
    uow = FakeUnitOfWork()
    service = DictionaryService(uow, _unused_resolver, _unused_index)

    assert await service.senses([]) == []
    assert uow.entries.calls == []


async def test_deleting_an_absent_entry_leaves_the_vector_index_alone():
    """No row removed means no vector to forget — and no pointless index write."""
    index = FakeVectorIndex()
    uow = FakeUnitOfWork(senses=FakeRepo(ids_for_word=[7]), words=FakeRepo(delete=False))
    service = DictionaryService(uow, _unused_resolver, index)

    assert await service.delete_entry(1) is False
    assert index.deleted == []


async def test_deleting_an_entry_forgets_exactly_its_own_sense_vectors():
    index = FakeVectorIndex()
    uow = FakeUnitOfWork(senses=FakeRepo(ids_for_word=[7, 9]), words=FakeRepo(delete=True))
    service = DictionaryService(uow, _unused_resolver, index)

    assert await service.delete_entry(1) is True
    assert index.deleted == [["7", "9"]]
    assert uow.commits == 1


async def test_a_broken_vector_index_never_fails_a_delete():
    """The vector store cannot join the delete's transaction, so it cannot veto it."""
    uow = FakeUnitOfWork(senses=FakeRepo(ids_for_word=[7]), words=FakeRepo(delete=True))
    service = DictionaryService(uow, _unused_resolver, BrokenVectorIndex())

    assert await service.delete_entry(1) is True


# --- tags -------------------------------------------------------------------


async def test_tag_writes_commit_once_each():
    uow = FakeUnitOfWork(tags=FakeRepo(rename=True, delete=True, merge=3))
    service = TagService(uow)

    assert await service.rename("cars", name="Cars") is True
    assert await service.delete("cars") is True
    assert await service.merge(["car", "auto"], "cars") == 3
    assert uow.commits == 3


async def test_merging_forwards_the_sources_as_a_list():
    """The port takes a list; a caller passing any sequence must still work."""
    uow = FakeUnitOfWork(tags=FakeRepo(merge=0))
    service = TagService(uow)

    await service.merge(("a", "b"), "c")

    assert uow.tags.calls[0] == ("merge", (["a", "b"], "c"), {})


# --- themes -----------------------------------------------------------------


async def test_a_theme_name_that_normalizes_to_nothing_is_rejected():
    service = ThemeService(FakeUnitOfWork(), _unused, _unused, _unused, _unused, 12)

    with pytest.raises(ValueError, match="no valid key"):
        await service.create("   ", "style", description="d", tone="t")


async def test_creating_a_theme_with_full_metadata_calls_no_model():
    """Supplying description and tone must skip the LLM expansion entirely."""

    def metadata():
        raise AssertionError("the metadata generator must not be called")

    uow = FakeUnitOfWork(themes=FakeRepo(create=_ThemeRecord()))
    service = ThemeService(uow, _unused, metadata, _unused, _unused, 12)

    await service.create("Pirate", "arrr", description="d", tone="t")

    assert uow.commits == 1


async def test_updating_an_unknown_theme_raises_rather_than_creating_one():
    uow = FakeUnitOfWork(themes=FakeRepo(update=None))
    service = ThemeService(uow, _unused, _unused, _unused, _unused, 12)

    with pytest.raises(ValueError, match="unknown theme"):
        await service.update("ghost", name="Ghost")


async def test_restyling_refuses_a_word_that_is_not_done():
    """Theming reads the neutral senses, so a pending word is a caller mistake."""

    async def status(_word_id):
        return "pending"

    service = ThemeService(FakeUnitOfWork(), _unused, _unused, _unused, status, 12)

    with pytest.raises(ValueError, match="not done"):
        await service.restyle_word(1, 2, "style")


async def test_appending_themed_examples_requires_an_existing_overlay():
    """Asking for examples must never theme the whole word as a side effect."""

    async def resolve(_theme):
        return (2, "style")

    uow = FakeUnitOfWork(
        senses=FakeRepo(example_context=(object(), [])),
        themes=FakeRepo(overlay_for_sense=None),
    )
    service = ThemeService(uow, _unused, _unused, _unused, _unused, 12)
    service.resolve_or_raise = resolve  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="no themed overlay"):
        await service.append_examples(5, 3, "pirate")


# --- search -----------------------------------------------------------------


async def test_semantic_search_is_empty_for_a_non_positive_k():
    service = SearchService(FakeUnitOfWork(), _unused, FakeEmbedder(), FakeVectorIndex())

    assert await service.semantic_search("q", k=0) == []
    assert await service.semantic_search("q", k=-1) == []


async def test_semantic_search_ranks_by_the_index_and_hydrates_from_the_store():
    index = FakeVectorIndex(hits=[VectorHit("9", 0.9), VectorHit("7", 0.5)])
    uow = FakeUnitOfWork(
        senses=FakeRepo(semantic_rows=lambda ids: [_sense_row(i, i * 10, f"w{i}") for i in ids])
    )
    service = SearchService(uow, _unused, FakeEmbedder(), index)

    hits = await service.semantic_search("q", k=2)

    assert [hit.score for hit in hits] == [0.9, 0.5]
    assert [hit.display for hit in hits] == ["w9", "w7"]
    # Only current-model vectors are eligible.
    assert index.queried[0][2] == {"model": "fake-model"}


async def test_semantic_search_asks_for_more_than_k_to_absorb_stale_vectors():
    """A vector whose sense is gone must not consume one of the caller's k slots."""
    index = FakeVectorIndex(hits=[])
    service = SearchService(FakeUnitOfWork(), _unused, FakeEmbedder(), index)

    await service.semantic_search("q", k=3)

    assert index.queried[0][1] > 3


async def test_semantic_search_raises_when_the_encoder_is_unavailable():
    """A broken encoder must not read as "no match" — that is a wrong answer."""

    class DeadEmbedder:
        model_name = "fake-model"

        async def embed_one(self, _text):
            raise RuntimeError("extra not installed")

    service = SearchService(FakeUnitOfWork(), _unused, DeadEmbedder(), FakeVectorIndex())

    with pytest.raises(RuntimeError, match="extra not installed"):
        await service.semantic_search("q")


async def test_semantic_search_raises_when_the_index_is_unreachable():
    service = SearchService(FakeUnitOfWork(), _unused, FakeEmbedder(), BrokenVectorIndex())

    with pytest.raises(RuntimeError):
        await service.semantic_search("q")


async def test_semantic_search_is_empty_when_the_index_holds_nothing():
    """The one legitimate empty answer: the search ran and matched nothing."""
    service = SearchService(FakeUnitOfWork(), _unused, FakeEmbedder(), FakeVectorIndex(hits=[]))

    assert await service.semantic_search("q") == []


# --- enrichment -------------------------------------------------------------


def _enrichment(uow, *, embedder=None, index=None, generator=None, judge=None):
    return EnrichmentService(
        uow,
        embedder or FakeEmbedder(),
        lambda: generator,
        lambda: judge,
        _unused,
        _unused,
        12,
        index or FakeVectorIndex(),
    )


async def test_adding_zero_examples_calls_no_model_and_writes_nothing():
    async def read_senses(sense_ids):
        return [SenseView(definition="d", tier="core", pos=None, cefr_level=None)]

    def generator():
        raise AssertionError("the generator must not be called for n <= 0")

    uow = FakeUnitOfWork(senses=FakeRepo(example_context=(object(), [])))
    service = EnrichmentService(
        uow,
        FakeEmbedder(),
        generator,
        lambda: None,
        read_senses,
        _unused,
        12,
        FakeVectorIndex(),
    )

    await service.add_examples(1, n=0)

    assert uow.commits == 0


async def test_embedding_skips_senses_the_index_already_holds():
    index = FakeVectorIndex(stored={"7"})
    uow = FakeUnitOfWork(senses=FakeRepo(needing_embedding=[_need(7), _need(9)]))
    service = _enrichment(uow, index=index)

    assert await service.embed_missing() == 1
    assert [record.id for record in index.upserted[0]] == ["9"]


async def test_embedding_raises_when_the_index_is_unreachable():
    """Zero must mean "nothing needed doing", so an unreachable index cannot return 0."""
    uow = FakeUnitOfWork(senses=FakeRepo(needing_embedding=[_need(7)]))

    with pytest.raises(RuntimeError):
        await _enrichment(uow, index=BrokenVectorIndex()).embed_missing()


async def test_a_failing_encoder_propagates_out_of_embedding():
    class DeadEmbedder:
        model_name = "fake-model"

        async def embed(self, _texts):
            raise RuntimeError("CUDA out of memory")

    uow = FakeUnitOfWork(senses=FakeRepo(needing_embedding=[_need(7)]))

    with pytest.raises(RuntimeError, match="CUDA"):
        await _enrichment(uow, embedder=DeadEmbedder()).embed_missing()


async def test_the_backfill_prunes_vectors_whose_sense_no_longer_exists():
    index = FakeVectorIndex(stored={"7", "4242"})
    uow = FakeUnitOfWork(
        senses=FakeRepo(live_sense_ids={7}, needing_embedding=[]),
    )

    await _enrichment(uow, index=index).backfill_embeddings()

    assert index.deleted == [["4242"]]


async def test_resolving_relations_without_a_judge_is_a_no_op():
    """No LLM configured degrades to doing nothing, like the other model paths."""
    uow = FakeUnitOfWork(senses=FakeRepo(pending_relations=[_task()]))

    assert await _enrichment(uow, judge=None).resolve_relations() == []


# --- shared fakes and helpers -----------------------------------------------


class FakeEmbedder:
    model_name = "fake-model"

    async def embed_one(self, _text):
        return [1.0, 0.0]

    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeVectorIndex:
    def __init__(self, hits=None, stored=None) -> None:
        self._hits = hits or []
        self._stored = set(stored or ())
        self.queried: list[tuple] = []
        self.upserted: list[list] = []
        self.deleted: list[list[str]] = []

    async def query(self, vector, k, where=None):
        self.queried.append((vector, k, where))
        return self._hits

    async def upsert(self, records):
        self.upserted.append(list(records))
        return len(records)

    async def delete(self, ids):
        self.deleted.append(list(ids))
        return len(ids)

    async def ids(self, where=None):
        return set(self._stored)

    async def fetch(self, ids):
        return {stored: [1.0, 0.0] for stored in ids if stored in self._stored}


class BrokenVectorIndex:
    """Every operation fails. Vectors are eventually consistent, so nothing may."""

    async def _boom(self, *_args, **_kwargs):
        raise RuntimeError("index unreachable")

    query = upsert = delete = ids = fetch = _boom


class _ThemeRecord:
    id = 1
    key = "pirate"
    name = "Pirate"
    style_prompt = "arrr"
    description = "d"
    tone = "t"


def _need(sense_id: int):
    from lexi_ai.domain.models import SenseEmbeddingNeed

    return SenseEmbeddingNeed(sense_id, f"w{sense_id}", "a definition")


def _task():
    from lexi_ai.domain.models import ResolveTask

    return ResolveTask(
        edge_id=1,
        rel_type="synonym",
        gloss="g",
        source_def="a definition",
        source_pos="noun",
        candidates=[],
    )


async def _unused_resolver(theme):  # noqa: ANN001, ANN202
    raise AssertionError("the theme resolver must not be called")


_unused_index = BrokenVectorIndex()


def _unused(*_args, **_kwargs):
    raise AssertionError("this collaborator must not be used")
