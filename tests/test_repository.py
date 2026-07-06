"""Tests for the persistence repository (Phase 5).

In-memory SQLite with a shared connection (StaticPool). Each test builds a
``GeneratedResult`` and asserts the write-path invariants: dedup/idempotency,
multi-unit split, stub-row linking, Cambridge-first cefr, and the error path.
"""

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from lexi_ai.db import create_session_factory, init_models, session_scope
from lexi_ai.generation.schemas import (
    GeneratedEntry,
    GeneratedResult,
)
from lexi_ai.models import EntryLink, Example, Sense, SenseReference, Word, WordAlias
from lexi_ai.normalize import match_key
from lexi_ai.persistence.repository import Repository


@pytest.fixture
async def session_factory():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _fk_on(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    await init_models(engine)
    yield create_session_factory(engine)
    await engine.dispose()


@pytest.fixture
def repo(session_factory):
    return Repository(session_factory)


def _color_result() -> GeneratedResult:
    return GeneratedResult(
        units=[
            GeneratedEntry(
                norm="color",
                entry_type="word",
                pos="noun",
                senses=[
                    {
                        "definition": "the property of reflecting light",
                        "tier": "core",
                        "cefr_level": "A1",
                        "examples": ["a bright color"],
                        "references": [{"source": "cambridge", "source_ref": "s1"}],
                    }
                ],
                aliases=[{"alias_norm": "colour", "type": "spelling_uk", "dialect": "uk"}],
                related=[{"norm": "hue", "rel_type": "synonym"}],
            )
        ]
    )


async def _count(session_factory, model) -> int:
    async with session_scope(session_factory) as session:
        return (await session.execute(select(func.count()).select_from(model))).scalar_one()


async def test_persist_creates_full_graph(repo, session_factory):
    words = await repo.persist_result(_color_result())
    assert len(words) == 1
    assert words[0].match_key == match_key("color")
    assert words[0].status == "done"
    assert await _count(session_factory, WordAlias) == 1
    assert await _count(session_factory, Sense) == 1
    assert await _count(session_factory, Example) == 1
    assert await _count(session_factory, SenseReference) == 1
    # 'hue' stub + the 'color' word = 2 words.
    assert await _count(session_factory, Word) == 2
    assert await _count(session_factory, EntryLink) == 1


async def test_persist_is_idempotent(repo, session_factory):
    await repo.persist_result(_color_result())
    await repo.persist_result(_color_result())
    # No duplicates across the board.
    assert await _count(session_factory, Word) == 2  # color + hue stub
    assert await _count(session_factory, WordAlias) == 1
    assert await _count(session_factory, Sense) == 1
    assert await _count(session_factory, Example) == 1
    assert await _count(session_factory, EntryLink) == 1


async def test_multi_unit_split_creates_n_words(repo, session_factory):
    result = GeneratedResult(
        units=[
            GeneratedEntry(
                norm="idiom one",
                entry_type="idiom",
                senses=[{"definition": "meaning one", "tier": "common"}],
            ),
            GeneratedEntry(
                norm="idiom two",
                entry_type="idiom",
                senses=[{"definition": "meaning two", "tier": "common"}],
            ),
        ]
    )
    words = await repo.persist_result(result)
    assert len(words) == 2
    keys = {w.match_key for w in words}
    assert keys == {match_key("idiom one"), match_key("idiom two")}


async def test_related_creates_pending_stub_and_real_link(repo, session_factory):
    await repo.persist_result(_color_result())
    async with session_scope(session_factory) as session:
        stub = (
            await session.execute(select(Word).where(Word.match_key == match_key("hue")))
        ).scalar_one()
        assert stub.status == "pending"
        link = (await session.execute(select(EntryLink))).scalar_one()
        assert link.to_word_id == stub.id  # real FK id, not a string
        assert link.rel_type == "synonym"


async def test_cambridge_cefr_overrides_llm(repo, session_factory):
    # LLM said B2, but Cambridge reference s1 carries A1 -> A1 must win.
    result = GeneratedResult(
        units=[
            GeneratedEntry(
                norm="book",
                entry_type="word",
                senses=[
                    {
                        "definition": "a written text",
                        "tier": "core",
                        "cefr_level": "B2",
                        "references": [{"source": "cambridge", "source_ref": "s1"}],
                    }
                ],
            )
        ]
    )
    await repo.persist_result(result, cambridge_cefr={"s1": "A1"})
    async with session_scope(session_factory) as session:
        sense = (await session.execute(select(Sense))).scalar_one()
        assert sense.cefr_level == "A1"


async def test_cambridge_cefr_matches_labelled_source_ref(repo, session_factory):
    # The prompt shows senses as "sense#42"; if the model echoes that labelled
    # form as source_ref, it must STILL resolve against a map keyed by bare id.
    # (Regression for the silent Cambridge-first CEFR fall-through.)
    result = GeneratedResult(
        units=[
            GeneratedEntry(
                norm="book",
                entry_type="word",
                senses=[
                    {
                        "definition": "a written text",
                        "tier": "core",
                        "cefr_level": "B2",
                        "references": [{"source": "cambridge", "source_ref": "sense#42"}],
                    }
                ],
            )
        ]
    )
    # Map keyed by the bare id, exactly as api._cefr_map builds it from a bundle.
    await repo.persist_result(result, cambridge_cefr={"42": "A1"})
    async with session_scope(session_factory) as session:
        sense = (await session.execute(select(Sense))).scalar_one()
        assert sense.cefr_level == "A1"


async def test_cefr_llm_fallback_when_no_cambridge(repo, session_factory):
    result = GeneratedResult(
        units=[
            GeneratedEntry(
                norm="book",
                entry_type="word",
                senses=[{"definition": "d", "tier": "core", "cefr_level": "B2"}],
            )
        ]
    )
    await repo.persist_result(result)  # no cambridge_cefr map
    async with session_scope(session_factory) as session:
        sense = (await session.execute(select(Sense))).scalar_one()
        assert sense.cefr_level == "B2"


async def test_upsert_updates_existing_word_in_place(repo, session_factory):
    await repo.persist_result(_color_result())
    # Re-persist with a changed definition + extra sense.
    updated = GeneratedResult(
        units=[
            GeneratedEntry(
                norm="color",
                entry_type="word",
                pos="noun",
                senses=[
                    {"definition": "updated meaning", "tier": "core"},
                    {"definition": "second meaning", "tier": "common"},
                ],
            )
        ]
    )
    await repo.persist_result(updated)
    async with session_scope(session_factory) as session:
        word = (
            await session.execute(select(Word).where(Word.match_key == match_key("color")))
        ).scalar_one()
        senses = (
            (
                await session.execute(
                    select(Sense).where(Sense.word_id == word.id).order_by(Sense.sense_order)
                )
            )
            .scalars()
            .all()
        )
        assert [s.definition for s in senses] == ["updated meaning", "second meaning"]
        # color + hue stub only; no duplicate color.
        assert await _count(session_factory, Word) == 2


async def test_stub_promoted_to_done_on_generation(repo, session_factory):
    # First create 'hue' as a pending stub (via color's related link).
    await repo.persist_result(_color_result())
    # Now generate 'hue' itself -> the same row flips to done.
    hue = GeneratedResult(
        units=[
            GeneratedEntry(
                norm="hue",
                entry_type="word",
                senses=[{"definition": "a color shade", "tier": "core"}],
            )
        ]
    )
    words = await repo.persist_result(hue)
    assert words[0].match_key == match_key("hue")
    assert words[0].status == "done"
    # Still only 2 words: no duplicate hue.
    assert await _count(session_factory, Word) == 2


async def test_error_path_sets_status_error(session_factory):
    # A repo whose reload step is fine, but force a failure mid-persist by
    # feeding a sense that violates NOT NULL via a monkeypatched normalize?
    # Simpler: make match_key raise for a poisoned norm through a subclass.
    class BoomRepo(Repository):
        async def _sync_senses(self, *a, **k):
            raise RuntimeError("boom during senses")

    repo = BoomRepo(session_factory)
    with pytest.raises(RuntimeError, match="boom"):
        await repo.persist_result(_color_result())

    async with session_scope(session_factory) as session:
        word = (
            await session.execute(select(Word).where(Word.match_key == match_key("color")))
        ).scalar_one_or_none()
        assert word is not None
        assert word.status == "error"
        assert "boom" in (word.error_msg or "")


async def test_get_done_keys(repo, session_factory):
    await repo.persist_result(_color_result())
    keys = await repo.get_done_keys()
    assert match_key("color") in keys
    # 'hue' is a pending stub, not done.
    assert match_key("hue") not in keys


# --- IPA (Phase 2: selective anchoring) -----------------------------------


def _ipa_result(ipa_uk: str | None, ipa_us: str | None) -> GeneratedResult:
    return GeneratedResult(
        units=[
            GeneratedEntry(
                norm="lead",
                entry_type="word",
                pos="noun",
                senses=[
                    {
                        "definition": "a heavy metal",
                        "tier": "core",
                        "ipa_uk": ipa_uk,
                        "ipa_us": ipa_us,
                    }
                ],
            )
        ]
    )


async def test_ipa_persisted_on_sense(repo, session_factory):
    await repo.persist_result(_ipa_result("led", "led"))
    async with session_scope(session_factory) as session:
        sense = (await session.execute(select(Sense))).scalar_one()
    assert sense.ipa_uk == "led"
    assert sense.ipa_us == "led"


async def test_ipa_nul_and_control_chars_survive_persist(repo, session_factory):
    # LLM-generated IPA for out-of-Cambridge words is untrusted: an embedded NUL
    # crashes the Postgres insert. It MUST route through _clean like every other
    # free LLM field, so persistence succeeds on both dialects (NUL stripped).
    await repo.persist_result(_ipa_result("l\x00ed", "l\ted"))
    async with session_scope(session_factory) as session:
        sense = (await session.execute(select(Sense))).scalar_one()
    assert "\x00" not in (sense.ipa_uk or "")
    assert "\t" not in (sense.ipa_us or "")


async def test_ipa_absent_is_none(repo, session_factory):
    await repo.persist_result(_ipa_result(None, None))
    async with session_scope(session_factory) as session:
        sense = (await session.execute(select(Sense))).scalar_one()
    assert sense.ipa_uk is None
    assert sense.ipa_us is None


async def test_insert_word_recovers_from_concurrent_duplicate(repo, session_factory):
    # Simulate a concurrent inserter: the key already exists as a committed row.
    # _insert_word must trip the UNIQUE constraint inside its savepoint, roll it
    # back cleanly, and adopt the existing row — WITHOUT poisoning the session
    # (which would surface as PendingRollbackError on the recovery SELECT).
    key = match_key("dup")
    async with session_scope(session_factory) as session:
        session.add(Word(norm="dup", match_key=key, status="done"))

    async with session_scope(session_factory) as session:
        word = await repo._insert_word(session, key, "dup", status="pending")
        # Adopted the pre-existing row (status stays 'done'), no new row.
        assert word.status == "done"

    assert await _count(session_factory, Word) == 1
