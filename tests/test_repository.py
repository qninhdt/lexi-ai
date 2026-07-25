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
from lexi_ai.domain.errors import StaleGenerationError
from lexi_ai.generation.schemas import (
    GeneratedEntry,
    GeneratedResult,
)
from lexi_ai.infrastructure.db.models import (
    Example,
    Sense,
    SenseReference,
    Word,
    WordAlias,
    WordRelation,
)
from lexi_ai.infrastructure.db.repositories.sense_repo import SqlSenseRepo
from lexi_ai.infrastructure.db.repositories.word_repo import SqlWordRepo
from lexi_ai.normalize import match_key
from tests.support.persistence_driver import PersistenceDriver


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
    return PersistenceDriver(session_factory)


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
                        "pos": "noun",
                        "cefr_level": "A1",
                        "examples": ["a bright color"],
                        "references": [{"source": "cambridge", "source_ref": "s1"}],
                    }
                ],
                aliases=[{"alias_norm": "colour", "type": "spelling_uk", "dialect": "uk"}],
                related=[{"norm": "hue", "rel_type": "word_family"}],
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
    assert await _count(session_factory, WordRelation) == 1


async def test_persist_is_idempotent(repo, session_factory):
    await repo.persist_result(_color_result())
    await repo.persist_result(_color_result())
    # No duplicates across the board.
    assert await _count(session_factory, Word) == 2  # color + hue stub
    assert await _count(session_factory, WordAlias) == 1
    assert await _count(session_factory, Sense) == 1
    assert await _count(session_factory, Example) == 1
    assert await _count(session_factory, WordRelation) == 1


async def test_multi_unit_split_creates_n_words(repo, session_factory):
    result = GeneratedResult(
        units=[
            GeneratedEntry(
                norm="idiom one",
                entry_type="idiom",
                senses=[{"definition": "meaning one", "tier": "common", "pos": "noun"}],
            ),
            GeneratedEntry(
                norm="idiom two",
                entry_type="idiom",
                senses=[{"definition": "meaning two", "tier": "common", "pos": "noun"}],
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
        link = (await session.execute(select(WordRelation))).scalar_one()
        assert link.to_word_id == stub.id  # real FK id, not a string
        assert link.rel_type == "word_family"


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
                        "pos": "noun",
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
                        "pos": "noun",
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
                senses=[{"definition": "d", "tier": "core", "cefr_level": "B2", "pos": "noun"}],
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
                    {"definition": "updated meaning", "tier": "core", "pos": "noun"},
                    {"definition": "second meaning", "tier": "common", "pos": "noun"},
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
                senses=[{"definition": "a color shade", "tier": "core", "pos": "noun"}],
            )
        ]
    )
    words = await repo.persist_result(hue)
    assert words[0].match_key == match_key("hue")
    assert words[0].status == "done"
    # Still only 2 words: no duplicate hue.
    assert await _count(session_factory, Word) == 2


async def test_error_path_sets_status_error(repo, session_factory, monkeypatch):
    # Force a failure mid-publish, after the word row exists but before the
    # transaction can commit, then assert the error status is still recorded. That
    # only works because error recording runs on an independent session.
    async def boom(*_args, **_kwargs):
        raise RuntimeError("boom during senses")

    monkeypatch.setattr(SqlSenseRepo, "sync", boom)
    with pytest.raises(RuntimeError, match="boom"):
        await repo.persist_result(_color_result())

    async with session_scope(session_factory) as session:
        word = (
            await session.execute(select(Word).where(Word.match_key == match_key("color")))
        ).scalar_one_or_none()
        assert word is not None
        assert word.status == "error"
        assert "boom" in (word.error_msg or "")


async def test_stale_generation_fence_cannot_publish_or_mark_newer_claim_as_error(
    repo, session_factory
):
    first = await repo.claim_generation("color")
    second = await repo.claim_generation("color")

    with pytest.raises(StaleGenerationError):
        await repo.persist_result(_color_result(), fence=first)

    async with session_scope(session_factory) as session:
        word = (
            await session.execute(select(Word).where(Word.match_key == match_key("color")))
        ).scalar_one()
        assert word.generation_epoch == second.epoch
        assert word.status == "pending"
        assert word.error_msg is None

    await repo.persist_result(_color_result(), fence=second)
    async with session_scope(session_factory) as session:
        word = (
            await session.execute(select(Word).where(Word.match_key == match_key("color")))
        ).scalar_one()
        assert word.status == "done"
        assert word.generation_epoch == second.epoch


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
                        "pos": "noun",
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


async def test_insert_word_recovers_from_concurrent_duplicate(session_factory):
    # Simulate a concurrent inserter: the key already exists as a committed row.
    # The insert must trip the UNIQUE constraint inside its savepoint, roll that
    # back cleanly, and adopt the existing row — WITHOUT poisoning the session
    # (which would surface as PendingRollbackError on the recovery SELECT).
    key = match_key("dup")
    async with session_scope(session_factory) as session:
        session.add(Word(norm="dup", match_key=key, status="done"))

    async with session_scope(session_factory) as session:
        word = await SqlWordRepo(session)._insert(key, "dup")
        # Adopted the pre-existing row (status stays 'done'), no new row.
        assert word.status == "done"

    assert await _count(session_factory, Word) == 1


# --- 1.3 untrusted-column NUL sanitation (dual-DB) ------------------------
#
# The neutral write path cleaned IPA/guideword/domain but NOT the neutral
# definition, examples, norm, alias_norm, or source_ref — all untrusted LM text
# landing in Postgres-strict columns. A NUL in any of them makes the Postgres
# INSERT raise; persist_result rolls back and marks the whole word
# status="error" (the fatal path). Invisible on SQLite, which accepts the NUL.
# Each test below is modeled on the IPA NUL test and confirmed to FAIL on
# pre-fix code (the raw value carried the NUL straight into the column).


def _nul_entry(
    *,
    norm: str = "shine",
    definition: str = "to give off light",
    example: str = "the stars shine",
    alias_norm: str | None = None,
    source_ref: str | None = None,
) -> GeneratedResult:
    sense: dict = {
        "definition": definition,
        "tier": "core",
        "pos": "verb",
        "examples": [example],
    }
    if source_ref is not None:
        sense["references"] = [{"source": "cambridge", "source_ref": source_ref}]
    entry_kwargs: dict = {
        "norm": norm,
        "entry_type": "word",
        "pos": "verb",
        "senses": [sense],
    }
    if alias_norm is not None:
        entry_kwargs["aliases"] = [{"alias_norm": alias_norm, "type": "spelling_uk"}]
    return GeneratedResult(units=[GeneratedEntry(**entry_kwargs)])


async def test_nul_in_definition_survives_persist(repo, session_factory):
    await repo.persist_result(_nul_entry(definition="to give off\x00 light"))
    async with session_scope(session_factory) as session:
        word = (await session.execute(select(Word))).scalar_one()
        sense = (await session.execute(select(Sense))).scalar_one()
    assert word.status == "done"  # not rolled back to "error"
    assert "\x00" not in sense.definition


async def test_nul_in_example_survives_persist(repo, session_factory):
    await repo.persist_result(_nul_entry(example="the stars\x00 shine"))
    async with session_scope(session_factory) as session:
        word = (await session.execute(select(Word))).scalar_one()
        ex = (await session.execute(select(Example))).scalar_one()
    assert word.status == "done"
    assert "\x00" not in ex.text


async def test_nul_in_norm_survives_persist(repo, session_factory):
    await repo.persist_result(_nul_entry(norm="shi\x00ne"))
    async with session_scope(session_factory) as session:
        word = (await session.execute(select(Word))).scalar_one()
    assert word.status == "done"
    assert "\x00" not in word.norm
    # match_key already strips control chars, so the key is NUL-free too.
    assert "\x00" not in word.match_key


async def test_nul_in_alias_norm_survives_persist(repo, session_factory):
    await repo.persist_result(_nul_entry(alias_norm="shi\x00ne up"))
    async with session_scope(session_factory) as session:
        word = (await session.execute(select(Word))).scalar_one()
        alias = (await session.execute(select(WordAlias))).scalar_one()
    assert word.status == "done"
    assert "\x00" not in alias.alias_norm
    assert "\x00" not in alias.alias_match_key


async def test_nul_in_source_ref_survives_persist(repo, session_factory):
    await repo.persist_result(_nul_entry(source_ref="s\x001"))
    async with session_scope(session_factory) as session:
        word = (await session.execute(select(Word))).scalar_one()
        ref = (await session.execute(select(SenseReference))).scalar_one()
    assert word.status == "done"
    assert "\x00" not in ref.source_ref


def test_over_length_source_ref_rejected_at_schema():
    # source_ref lands in String(255); an over-length value is the same dual-DB
    # class as the keys. Bound it at the model boundary (max_length=255) so it is
    # rejected before it can reach the DB and raise "value too long".
    import pydantic

    from lexi_ai.generation.schemas import GeneratedReference

    GeneratedReference(source="cambridge", source_ref="4" * 255)  # at the bound: ok
    with pytest.raises(pydantic.ValidationError):
        GeneratedReference(source="cambridge", source_ref="4" * 256)
