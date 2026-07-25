"""Per-aggregate repository behavior.

Splitting one god repository into aggregate-scoped ones moved a lot of query logic.
These tests pin the parts a caller depends on and that a rewrite could plausibly
get wrong: which population a read describes, what ordering it guarantees, and
that nothing returns a live ORM row across the boundary.
"""

import dataclasses

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from lexi_ai.db import create_session_factory, init_models
from lexi_ai.domain.models import ThemeRecord, WordRecord
from lexi_ai.generation.schemas import GeneratedEntry, GeneratedResult
from lexi_ai.infrastructure.db.uow import SqlAlchemyUnitOfWork
from lexi_ai.normalize import match_key, render
from tests.support.persistence_driver import PersistenceDriver


@pytest.fixture
async def session_factory():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _fk_on(dbapi_conn, _record):  # noqa: ANN001
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    await init_models(engine)
    yield create_session_factory(engine)
    await engine.dispose()


@pytest.fixture
def uow(session_factory):
    def _factory() -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork(session_factory)

    return _factory


@pytest.fixture
def repo(session_factory):
    return PersistenceDriver(session_factory)


def _entry(norm: str, *, topics=(), senses=None) -> GeneratedEntry:
    return GeneratedEntry(
        norm=norm,
        entry_type="word",
        senses=senses or [{"definition": f"meaning of {norm}", "tier": "core", "pos": "noun"}],
        topics=[{"tag": tag, "title": title} for tag, title in topics],
    )


async def _publish(repo, *entries: GeneratedEntry):
    return await repo.persist_result(GeneratedResult(units=list(entries)))


# --- words ------------------------------------------------------------------


async def test_words_returns_detached_records_not_orm_rows(repo, uow):
    await _publish(repo, _entry("stone"))

    async with uow() as work:
        listing = await work.words.listing()
        record = await work.words.record(listing[0].word_id)

    # Readable after the transaction closed, which an ORM row would not reliably be.
    assert isinstance(record, WordRecord)
    assert dataclasses.asdict(record)["norm"] == "stone"
    assert record.match_key == match_key("stone")
    assert record.status == "done"


async def test_word_listing_is_norm_sorted_and_filtered_by_status(repo, uow):
    await _publish(repo, _entry("beta"), _entry("alpha"))

    async with uow() as work:
        done = await work.words.listing()
        pending = await work.words.listing(status="pending")

    assert [row.norm for row in done] == ["alpha", "beta"]
    assert pending == []


async def test_word_listing_paginates(repo, uow):
    await _publish(repo, _entry("one"), _entry("two"), _entry("three"))

    async with uow() as work:
        first = await work.words.listing(limit=2)
        second = await work.words.listing(limit=2, offset=2)

    assert len(first) == 2 and len(second) == 1
    assert {row.norm for row in first} & {row.norm for row in second} == set()


async def test_done_keys_reports_only_generated_words(repo, uow):
    await _publish(repo, _entry("kept"))
    async with uow() as work:
        await work.words.get_or_create("just-a-stub")
        await work.commit()

    async with uow() as work:
        keys = await work.words.done_keys()

    assert keys == {match_key("kept")}


async def test_deleting_a_word_cascades_its_senses(repo, uow):
    await _publish(repo, _entry("temporary"))

    async with uow() as work:
        word_id = (await work.words.listing())[0].word_id
        assert await work.words.delete(word_id) is True
        await work.commit()

    async with uow() as work:
        assert await work.words.listing() == []
        assert await work.senses.live_sense_ids() == set()


async def test_deleting_an_unknown_word_reports_no_row_removed(uow):
    async with uow() as work:
        assert await work.words.delete(4242) is False


async def test_get_or_create_is_idempotent_per_key(uow):
    async with uow() as work:
        first = await work.words.get_or_create("Recurring")
        second = await work.words.get_or_create("recurring")
        await work.commit()

    assert first == second


# --- tags -------------------------------------------------------------------


async def test_tag_reads_describe_only_the_done_population(repo, uow):
    """All three tag reads must agree on which words count.

    They previously joined through to words independently; if one drifts, the
    vocabulary injected into prompts stops matching the browse surface.
    """
    await _publish(repo, _entry("bank", topics=[("finance", "Finance")]))

    async with uow() as work:
        names = await work.tags.names()
        usage = await work.tags.usage()
        words = await work.tags.words_for_key("finance")

    assert [row.name for row in names] == ["finance"]
    assert [(row.name, row.count) for row in usage] == [("finance", 1)]
    assert [row.norm for row in words] == ["bank"]


async def test_tag_usage_is_ordered_by_count_then_name(repo, uow):
    await _publish(
        repo,
        _entry("one", topics=[("common", "Common"), ("rare", "Rare")]),
        _entry("two", topics=[("common", "Common")]),
    )

    async with uow() as work:
        usage = await work.tags.usage()

    assert [(row.name, row.count) for row in usage] == [("common", 2), ("rare", 1)]


async def test_renaming_a_tag_keeps_its_identity(repo, uow):
    await _publish(repo, _entry("word", topics=[("topic", "Topic")]))

    async with uow() as work:
        assert await work.tags.rename("topic", title="Renamed Topic") is True
        await work.commit()

    async with uow() as work:
        # Still addressable by the same key: renaming display text never re-keys.
        assert [row.title for row in await work.tags.usage()] == ["Renamed Topic"]
        assert len(await work.tags.words_for_key("topic")) == 1


async def test_renaming_requires_something_to_change(uow):
    async with uow() as work:
        with pytest.raises(ValueError, match="name and/or title"):
            await work.tags.rename("topic")


async def test_merging_folds_members_and_drops_the_source(repo, uow):
    await _publish(
        repo,
        _entry("a", topics=[("source", "Source")]),
        _entry("b", topics=[("target", "Target")]),
    )

    async with uow() as work:
        moved = await work.tags.merge(["source"], "target")
        await work.commit()

    assert moved == 1
    async with uow() as work:
        assert [row.name for row in await work.tags.names()] == ["target"]
        assert len(await work.tags.words_for_key("target")) == 2


async def test_merging_never_invents_the_destination(repo, uow):
    await _publish(repo, _entry("a", topics=[("source", "Source")]))

    async with uow() as work:
        with pytest.raises(ValueError, match="unknown destination tag"):
            await work.tags.merge(["source"], "absent")


async def test_merging_drops_a_duplicate_link_rather_than_re_pointing_it(repo, uow):
    """A word already carrying both tags would collide on the unique pair."""
    await _publish(repo, _entry("both", topics=[("source", "Source"), ("target", "Target")]))

    async with uow() as work:
        moved = await work.tags.merge(["source"], "target")
        await work.commit()

    assert moved == 0
    async with uow() as work:
        assert [(row.name, row.count) for row in await work.tags.usage()] == [("target", 1)]


# --- themes -----------------------------------------------------------------


async def test_theme_reads_return_records(uow):
    async with uow() as work:
        created = await work.themes.create("Pirate", "speak like a pirate")
        await work.commit()

    assert isinstance(created, ThemeRecord)
    assert created.key == "pirate"

    async with uow() as work:
        fetched = await work.themes.get("pirate")
        listed = await work.themes.list_all()

    assert fetched == created
    assert listed == [created]


async def test_creating_a_theme_twice_resolves_to_the_same_row(uow):
    async with uow() as work:
        first = await work.themes.create("Noir", "hard-boiled")
        second = await work.themes.create("noir", "ignored without overwrite")
        await work.commit()

    assert first.id == second.id
    assert second.style_prompt == "hard-boiled"


async def test_overwriting_a_theme_replaces_its_fields(uow):
    async with uow() as work:
        await work.themes.create("Noir", "hard-boiled")
        updated = await work.themes.create("Noir", "rewritten", overwrite=True)
        await work.commit()

    assert updated.style_prompt == "rewritten"


async def test_updating_an_unknown_theme_reports_a_miss(uow):
    async with uow() as work:
        assert await work.themes.update("absent", name="x") is None


async def test_updating_a_theme_never_re_keys_it(uow):
    async with uow() as work:
        await work.themes.create("Original", "prompt")
        updated = await work.themes.update("original", name="Renamed")
        await work.commit()

    assert updated.name == "Renamed"
    assert updated.key == "original"


async def test_resolving_a_theme_accepts_a_key_or_an_id(uow):
    async with uow() as work:
        created = await work.themes.create("Steampunk", "brass and steam")
        await work.commit()

    async with uow() as work:
        by_key = await work.themes.resolve("steampunk")
        by_id = await work.themes.resolve(created.id)
        missing = await work.themes.resolve("absent")

    assert by_key == (created.id, "brass and steam")
    assert by_id == by_key
    assert missing is None


async def test_deleting_a_theme_reports_whether_a_row_went(uow):
    async with uow() as work:
        await work.themes.create("Doomed", "prompt")
        await work.commit()

    async with uow() as work:
        assert await work.themes.delete("doomed") is True
        assert await work.themes.delete("doomed") is False
        await work.commit()


# --- senses -----------------------------------------------------------------


async def test_senses_needing_embedding_covers_only_done_words(repo, uow):
    await _publish(repo, _entry("vector"))

    async with uow() as work:
        pending = await work.senses.needing_embedding()

    assert [row.norm for row in pending] == ["vector"]
    assert pending[0].definition == "meaning of vector"


async def test_embedding_candidates_ignore_the_vector_store(repo, uow):
    """This store no longer knows what is embedded, so every done sense is offered.

    Which of them actually needs work is the vector index's answer, subtracted by
    the caller — keeping one source of truth for what is indexed.
    """
    await _publish(repo, _entry("vector"))

    async with uow() as work:
        first = await work.senses.needing_embedding()
        second = await work.senses.needing_embedding()

    assert [row.sense_id for row in first] == [row.sense_id for row in second]


async def test_semantic_rows_preserve_rank_order_and_skip_vanished_senses(repo, uow):
    await _publish(repo, _entry("vector"))

    async with uow() as work:
        sense_id = (await work.senses.needing_embedding())[0].sense_id
        # 999 is a stale vector id: it drops out instead of raising.
        rows = await work.senses.semantic_rows([999, sense_id])

    assert [row.sense_id for row in rows] == [sense_id]
    assert rows[0].norm == "vector"
    assert rows[0].definition == "meaning of vector"


async def test_sense_ids_for_word_lists_every_sense_of_that_word(repo, uow):
    await _publish(repo, _entry("vector"))

    async with uow() as work:
        word_id = (await work.words.listing())[0].word_id
        assert await work.senses.ids_for_word(word_id) == sorted(await work.senses.live_sense_ids())
        assert await work.senses.ids_for_word(4242) == []


async def test_senses_for_theming_are_ordered_for_stable_prompt_numbering(repo, uow):
    """The caller pairs themed results back by position, so order is a contract."""
    await _publish(
        repo,
        _entry(
            "layered",
            senses=[
                {"definition": "first", "tier": "core", "pos": "noun"},
                {"definition": "second", "tier": "common", "pos": "noun"},
            ],
        ),
    )

    async with uow() as work:
        word_id = (await work.words.listing())[0].word_id
        senses = await work.senses.for_theming(word_id)

    assert [row.definition for row in senses] == ["first", "second"]
    assert [row.tier for row in senses] == ["core", "common"]


async def test_word_id_for_an_unknown_sense_raises(uow):
    from sqlalchemy.exc import NoResultFound

    async with uow() as work:
        with pytest.raises(NoResultFound):
            await work.senses.word_id_for(999)


async def test_example_context_reports_a_miss_for_an_unknown_sense(uow):
    async with uow() as work:
        assert await work.senses.example_context(999) is None


async def test_appending_examples_continues_the_existing_order(repo, uow):
    await _publish(
        repo,
        _entry(
            "listed",
            senses=[
                {
                    "definition": "d",
                    "tier": "core",
                    "pos": "noun",
                    "examples": ["first example"],
                }
            ],
        ),
    )

    async with uow() as work:
        sense_id = (await work.senses.needing_embedding())[0].sense_id
        inserted = await work.senses.append_examples(sense_id, ["second example", "  "])
        await work.commit()

    # The blank text is skipped rather than stored empty.
    assert inserted == 1
    async with uow() as work:
        _context, examples = await work.senses.example_context(sense_id)
    assert examples == ["first example", "second example"]


# --- stats ------------------------------------------------------------------


async def test_stats_counts_every_aggregate_in_one_snapshot(repo, uow):
    await _publish(repo, _entry("counted", topics=[("topic", "Topic")]))
    async with uow() as work:
        await work.themes.create("Theme", "prompt")
        await work.commit()

    async with uow() as work:
        snapshot = await work.stats.snapshot()

    assert snapshot.words_by_status == {"done": 1}
    assert snapshot.senses == 1
    assert snapshot.tags == 1
    assert snapshot.themes == 1
    assert snapshot.themed_words == 0
    assert snapshot.questions == 0


# --- lookup reads backing search --------------------------------------------


async def test_generated_by_cambridge_returns_the_stored_lemma_not_a_display(repo, uow):
    """The row carries the raw lemma; rendering is the caller's job.

    This distinction is easy to lose. A lemma keeps placeholders like ``{sb}`` that a
    caller must never show, so search renders before building a hit. A reader that
    forwarded ``norm`` straight to ``display`` would leak the placeholder, and most
    words look identical either way, so nothing else would notice.
    """
    entry = GeneratedEntry(
        norm="ask {sb} out",
        entry_type="phrase",
        senses=[{"definition": "invite someone", "tier": "core", "pos": "verb"}],
    )
    await repo.persist_result(GeneratedResult(units=[entry]), cambridge_word_id=4242)

    async with uow() as work:
        found = await work.words.generated_by_cambridge([4242])

    assert found[4242].norm == "ask {sb} out"
    assert render(found[4242].norm) == "ask somebody out"


async def test_resolve_key_matches_a_headword_and_an_alias(repo, uow):
    await _publish(
        repo,
        GeneratedEntry(
            norm="color",
            entry_type="word",
            senses=[{"definition": "hue", "tier": "core", "pos": "noun"}],
            aliases=[{"alias_norm": "colour", "type": "spelling_uk", "dialect": "uk"}],
        ),
    )

    async with uow() as work:
        by_headword = await work.words.resolve_key(match_key("color"))
        by_alias = await work.words.resolve_key(match_key("colour"))

    assert [row.status for row in by_headword] == ["done"]
    assert by_alias == by_headword  # the alias resolves to the same word


async def test_norm_and_cambridge_reports_provenance(repo, uow):
    await repo.persist_result(GeneratedResult(units=[_entry("anchored")]), cambridge_word_id=77)

    async with uow() as work:
        word_id = (await work.words.listing())[0].word_id
        norm, cambridge_id = await work.words.norm_and_cambridge(word_id)

    assert (norm, cambridge_id) == ("anchored", 77)
