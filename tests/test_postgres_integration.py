"""Opt-in dual-DB regression tier — real Postgres over ``LEXI_TEST_PG_URL``.

These tests prove the defect class the SQLite suite cannot see: NUL rejection,
``VARCHAR(n)`` length enforcement, and tz-aware datetime binding on
``TIMESTAMP WITHOUT TIME ZONE``. Each dual-DB regression is written on Phase 0's
harness so it can be confirmed RED on pre-fix code and green after.

Skips cleanly with no driver / no URL — the default ``uv run pytest`` stays
hermetic. The ``pg_session_factory`` fixture is auto-discovered from
``conftest.py``; see it for the install + run invocation.

Phase 4 regressions (all five untrusted columns + H2 datetime bind):
  - NUL in definition, example, norm, alias_norm, source_ref -> word stays done
  - Over-length source_ref -> clean truncation, not a DB "value too long" raise
  - _mark_unresolvable stamps naive UTC -> edge leaves pending (no re-judge loop)
"""

import pytest
from sqlalchemy import select

from lexi_ai.db import session_scope
from lexi_ai.domain.errors import StaleGenerationError
from lexi_ai.domain.models import ResolveDecision
from lexi_ai.generation.schemas import (
    GeneratedAlias,
    GeneratedEntry,
    GeneratedReference,
    GeneratedResult,
    GeneratedSense,
)
from lexi_ai.infrastructure.db.models import (
    Example,
    Sense,
    SenseReference,
    SenseRelation,
    Word,
    WordAlias,
)
from lexi_ai.normalize import match_key
from tests.conftest import requires_postgres
from tests.support.persistence_driver import PersistenceDriver

pytestmark = requires_postgres


# ---------------------------------------------------------------------------
# Harness smoke test (Phase 0)
# ---------------------------------------------------------------------------


async def test_harness_connects_and_roundtrips_a_word(pg_session_factory):
    """The fixture applies the domain Alembic baseline and a trivial ``Word``
    round-trips, proving the harness before any regression leans on it."""
    async with session_scope(pg_session_factory) as session:
        session.add(Word(norm="book", match_key=match_key("book"), status="done"))

    async with session_scope(pg_session_factory) as session:
        row = (await session.execute(select(Word).where(Word.norm == "book"))).scalar_one()
        assert row.match_key == "book"
    assert row.status == "done"


async def test_postgres_generation_epoch_rejects_a_stale_worker_publish(pg_session_factory):
    """Two independently constructed repositories fence stale destructive writes."""
    older, newer = PersistenceDriver(pg_session_factory), PersistenceDriver(pg_session_factory)
    first_claim = await older.claim_generation("shine")
    second_claim = await newer.claim_generation("shine")

    with pytest.raises(StaleGenerationError):
        await older.persist_result(_pg_entry(), fence=first_claim)
    await newer.persist_result(_pg_entry(), fence=second_claim)

    async with session_scope(pg_session_factory) as session:
        word = (await session.execute(select(Word).where(Word.match_key == "shine"))).scalar_one()
    assert word.status == "done"
    assert word.generation_epoch == second_claim.epoch


# ---------------------------------------------------------------------------
# Shared helper: build a GeneratedResult with one word + one sense
# ---------------------------------------------------------------------------


def _pg_entry(
    *,
    norm: str = "shine",
    definition: str = "to give off light",
    example: str = "the stars shine",
    alias_norm: str | None = None,
) -> GeneratedResult:
    """Minimal GeneratedResult for Postgres NUL/length regression tests.

    source_ref is NOT parameterized here: a NUL/over-length source_ref must be
    injected past Pydantic's ``max_length=255`` boundary, so those two tests build
    their own entry with ``object.__setattr__`` rather than through this helper.
    """
    sense = GeneratedSense(
        definition=definition,
        tier="core",
        pos="verb",
        examples=[example],
        references=[],
    )
    aliases = []
    if alias_norm is not None:
        aliases = [GeneratedAlias(alias_norm=alias_norm, type="spelling_uk")]
    entry = GeneratedEntry(
        norm=norm,
        entry_type="word",
        pos="verb",
        senses=[sense],
        aliases=aliases,
    )
    return GeneratedResult(units=[entry])


def _sense_with_raw_source_ref(norm: str, source_ref: str) -> GeneratedResult:
    """A one-word result whose sole reference carries ``source_ref`` verbatim,
    bypassing the schema's ``max_length=255`` so the DB-side clean is exercised."""
    sense = GeneratedSense(
        definition="to shine",
        tier="core",
        pos="verb",
        examples=["it shines"],
        references=[GeneratedReference(source="cambridge", source_ref="placeholder")],
    )
    object.__setattr__(sense.references[0], "source_ref", source_ref)
    entry = GeneratedEntry(norm=norm, entry_type="word", pos="verb", senses=[sense])
    return GeneratedResult(units=[entry])


# ---------------------------------------------------------------------------
# Phase 4 — H2: tz-aware datetime bind (Postgres DataError detection)
# ---------------------------------------------------------------------------


async def test_mark_unresolvable_no_rejudge_loop_on_postgres(pg_session_factory):
    """H2 (dual-DB): _mark_unresolvable must stamp a NAIVE datetime so asyncpg
    does not raise DataError binding an aware value to TIMESTAMP WITHOUT TIME ZONE.

    The pre-fix bug: the swallowed DataError at apply_resolutions converted every
    unresolvable edge to state='error' WITHOUT stamping resolve_attempted_at, so
    the edge stayed derived-pending and was re-fetched forever. This test observes
    the post-fix downstream: after judging 'unresolvable', the edge leaves the
    pending predicate (resolve_attempted_at is stamped) — no re-queue.

    Confirmed to FAIL on pre-fix code: the swallowed DataError meant
    resolve_attempted_at was never written, so the edge remained pending.
    """
    repo = PersistenceDriver(pg_session_factory)
    # Plant a stub word + a pending self-relation to act as the WSD edge.
    async with session_scope(pg_session_factory) as session:
        word = Word(norm="glow", match_key=match_key("glow"), status="done")
        session.add(word)
        await session.flush()
        sense = Sense(word_id=word.id, definition="to emit light", tier="core", sense_order=0)
        session.add(sense)
        await session.flush()
        edge = SenseRelation(
            from_sense_id=sense.id,
            to_word_id=word.id,
            rel_type="synonym",
            gloss="emit light",
        )
        session.add(edge)
        await session.flush()
        edge_id = edge.id

    # Apply an 'unresolvable' decision (to_sense_id=None).
    outcomes = await repo.apply_resolutions(
        [ResolveDecision(edge_id=edge_id, to_sense_id=None, target_hash=None)]
    )
    assert outcomes[0].state in ("unresolvable", "noop"), f"unexpected state: {outcomes[0].state}"

    # The key assertion: resolve_attempted_at must be stamped (not None) and NAIVE.
    async with session_scope(pg_session_factory) as session:
        edge = (
            await session.execute(select(SenseRelation).where(SenseRelation.id == edge_id))
        ).scalar_one()

    assert edge.resolve_attempted_at is not None, (
        "resolve_attempted_at was not stamped — the DataError was likely swallowed "
        "and the edge would be re-judged forever (H2 regression)"
    )
    assert edge.resolve_attempted_at.tzinfo is None, (
        "resolve_attempted_at is tz-aware — asyncpg would have raised DataError "
        "binding it to TIMESTAMP WITHOUT TIME ZONE (H2 regression)"
    )


# ---------------------------------------------------------------------------
# Phase 4 — NUL in all five untrusted write columns (dual-DB confirmation)
# ---------------------------------------------------------------------------
#
# SQLite accepts NUL in text columns; Postgres rejects 0x00 in TEXT/VARCHAR.
# Each test below persists a value containing NUL and asserts:
#   (a) the word stays done (not rolled back to 'error')
#   (b) the stored value is NUL-free (the cleaner stripped it)
#
# Confirmed to FAIL on pre-fix code: the raw NUL reached the Postgres INSERT,
# raised, and _record_error flipped the word to status='error'.

_NUL_ROLLED_BACK = "NUL rolled the word back to error (H-1.3 Postgres regression)"


async def test_postgres_nul_in_definition_survives_persist(pg_session_factory):
    """NUL in neutral definition is stripped before the Postgres INSERT (1.3)."""
    repo = PersistenceDriver(pg_session_factory)
    await repo.persist_result(_pg_entry(definition="to give off\x00 light"))
    async with session_scope(pg_session_factory) as session:
        word = (await session.execute(select(Word))).scalar_one()
        sense = (await session.execute(select(Sense))).scalar_one()
    assert word.status == "done", _NUL_ROLLED_BACK
    assert "\x00" not in sense.definition


async def test_postgres_nul_in_example_survives_persist(pg_session_factory):
    """NUL in neutral example is stripped before the Postgres INSERT (1.3)."""
    repo = PersistenceDriver(pg_session_factory)
    await repo.persist_result(_pg_entry(example="the stars\x00 shine"))
    async with session_scope(pg_session_factory) as session:
        word = (await session.execute(select(Word))).scalar_one()
        ex = (await session.execute(select(Example))).scalar_one()
    assert word.status == "done", _NUL_ROLLED_BACK
    assert "\x00" not in ex.text


async def test_postgres_nul_in_norm_survives_persist(pg_session_factory):
    """NUL in norm is stripped before the Postgres INSERT (1.3 + 1.4)."""
    repo = PersistenceDriver(pg_session_factory)
    await repo.persist_result(_pg_entry(norm="shi\x00ne"))
    async with session_scope(pg_session_factory) as session:
        word = (await session.execute(select(Word))).scalar_one()
    assert word.status == "done", _NUL_ROLLED_BACK
    assert "\x00" not in word.norm
    assert "\x00" not in word.match_key


async def test_postgres_nul_in_alias_norm_survives_persist(pg_session_factory):
    """NUL in alias_norm is stripped before the Postgres INSERT (1.3)."""
    repo = PersistenceDriver(pg_session_factory)
    await repo.persist_result(_pg_entry(alias_norm="shi\x00ne up"))
    async with session_scope(pg_session_factory) as session:
        word = (await session.execute(select(Word))).scalar_one()
        alias = (await session.execute(select(WordAlias))).scalar_one()
    assert word.status == "done", _NUL_ROLLED_BACK
    assert "\x00" not in alias.alias_norm
    assert "\x00" not in alias.alias_match_key


async def test_postgres_nul_in_source_ref_survives_persist(pg_session_factory):
    """NUL in source_ref is stripped before the Postgres INSERT (1.3)."""
    repo = PersistenceDriver(pg_session_factory)
    await repo.persist_result(_sense_with_raw_source_ref("radiate", "s\x001"))
    async with session_scope(pg_session_factory) as session:
        word = (await session.execute(select(Word))).scalar_one()
        ref = (await session.execute(select(SenseReference))).scalar_one()
    assert word.status == "done", _NUL_ROLLED_BACK
    assert "\x00" not in ref.source_ref


# ---------------------------------------------------------------------------
# Phase 4 — over-length source_ref (Postgres VARCHAR(255) enforcement)
# ---------------------------------------------------------------------------


async def test_postgres_over_length_source_ref_is_clamped(pg_session_factory):
    """source_ref > 255 chars is clamped at persist time (1.3 + schema max_length).

    Pre-fix: the raw over-length value reached String(255) on Postgres and raised
    'value too long for type character varying(255)'. Post-fix: the value is
    clamped to _MAX_SOURCE_REF (255) by _clean() before the INSERT.
    """
    repo = PersistenceDriver(pg_session_factory)
    await repo.persist_result(_sense_with_raw_source_ref("gleam", "x" * 300))
    async with session_scope(pg_session_factory) as session:
        word = (await session.execute(select(Word))).scalar_one()
        ref = (await session.execute(select(SenseReference))).scalar_one()
    assert word.status == "done", "over-length source_ref rolled the word back to error"
    assert len(ref.source_ref) <= 255


# ---------------------------------------------------------------------------
# Schema access paths
# ---------------------------------------------------------------------------


async def test_senses_word_id_has_an_index_to_read_a_word_by(pg_session_factory):
    """`senses.word_id` is the most-filtered column and needs its own index.

    Unindexed, every read of a word's senses and the join behind entry assembly
    has no access path but a sequential scan of the whole table. The other
    `word_id` columns in the schema do not need one: aliases and tags lead their
    UNIQUE constraints with it, and Postgres backs a unique constraint with a btree
    index a lookup on the leading column can already use. `senses` has no such
    constraint.

    Asserted against a schema Alembic built, which is the point — the ORM
    declaration is not in question, whether the migration puts it in the database
    is. Dropping the `create_index` makes this fail, and the SQLite tier cannot
    tell either way.
    """
    from sqlalchemy import inspect

    from tests.conftest import PG_SCHEMA

    async with pg_session_factory() as session:
        connection = await session.connection()
        indexes = await connection.run_sync(
            lambda sync_connection: inspect(sync_connection).get_indexes(
                "senses", schema=PG_SCHEMA
            )
        )

    covering = [
        index
        for index in indexes
        if index["column_names"] and index["column_names"][0] == "word_id"
    ]
    assert covering, (
        "no index leads with senses.word_id, so reading a word's senses is a "
        f"sequential scan. Indexes present: {[i['name'] for i in indexes]}"
    )
