"""Transaction boundaries around the unit of work.

Phase 2 moved persistence behind aggregate repositories sharing one session. The
risk that introduced is not "does a write land" — the aggregate tests cover that —
but whether the boundaries still sit where they have to. Three patterns are
load-bearing and each fails in a specific, quiet way if folded into the wrong
transaction:

* The generation claim must COMMIT before provider work, or a competing worker
  cannot see the epoch and the fence stops fencing.
* A publish must be all-or-nothing across every unit of a result.
* Error recording must survive the rollback that triggered it, which means an
  independent session.
"""

import pytest
from sqlalchemy import func, select

from lexi_ai.application.generation_writer import GenerationWriter
from lexi_ai.config import Settings
from lexi_ai.db import create_engine, create_session_factory, init_models, session_scope
from lexi_ai.generation.schemas import GeneratedEntry, GeneratedResult
from lexi_ai.infrastructure.db.models import Sense, Word
from lexi_ai.infrastructure.db.repositories.sense_repo import SqlSenseRepo
from lexi_ai.infrastructure.db.uow import SqlAlchemyUnitOfWork
from lexi_ai.normalize import match_key


@pytest.fixture
async def engine(tmp_path):
    """A file-backed database built through the production engine factory.

    Two deliberate differences from the rest of the suite, both required to observe
    a transaction at all. First, a file rather than shared in-memory: the other
    fixtures pin one connection through StaticPool, so the "other" session joins the
    writer's transaction and uncommitted rows look committed. Second, the real
    :func:`create_engine`, because the transactional pragmas it installs are part of
    what these tests check.
    """
    engine = create_engine(Settings(db_url=f"sqlite+aiosqlite:///{tmp_path / 'uow.db'}"))
    await init_models(engine)
    yield engine
    await engine.dispose()


@pytest.fixture
def session_factory(engine):
    return create_session_factory(engine)


@pytest.fixture
def uow_factory(session_factory):
    def _factory() -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork(session_factory)

    return _factory


def _two_unit_result() -> GeneratedResult:
    return GeneratedResult(
        units=[
            GeneratedEntry(
                norm="light",
                entry_type="word",
                senses=[{"definition": "not heavy", "tier": "core", "pos": "adjective"}],
            ),
            GeneratedEntry(
                norm="lightly",
                entry_type="word",
                senses=[{"definition": "with little force", "tier": "core", "pos": "adverb"}],
            ),
        ]
    )


async def _count(session_factory, model) -> int:
    async with session_scope(session_factory) as session:
        return (await session.execute(select(func.count(model.id)))).scalar_one()


# --- commit boundary --------------------------------------------------------


async def test_work_is_invisible_until_the_unit_of_work_commits(session_factory, uow_factory):
    async with uow_factory() as uow:
        await uow.words.get_or_create("pending-visibility")
        await uow.flush()
        # A flush pushes the INSERT but does not end the transaction, so a separate
        # session must not see the row yet.
        assert await _count(session_factory, Word) == 0
        await uow.commit()

    assert await _count(session_factory, Word) == 1


async def test_leaving_the_unit_of_work_without_committing_discards_the_write(
    session_factory, uow_factory
):
    async with uow_factory() as uow:
        await uow.words.get_or_create("never-committed")
        await uow.flush()

    assert await _count(session_factory, Word) == 0


async def test_an_exception_rolls_back_every_write_in_the_unit_of_work(
    session_factory, uow_factory
):
    with pytest.raises(RuntimeError, match="halfway"):
        async with uow_factory() as uow:
            await uow.words.get_or_create("first")
            await uow.words.get_or_create("second")
            await uow.flush()
            raise RuntimeError("halfway through the use case")

    assert await _count(session_factory, Word) == 0


# --- the publish is atomic across units -------------------------------------


async def test_publish_commits_every_unit_together(session_factory, uow_factory):
    records = await GenerationWriter(uow_factory).publish(_two_unit_result())

    assert [record.norm for record in records] == ["light", "lightly"]
    assert await _count(session_factory, Word) == 2
    assert await _count(session_factory, Sense) == 2


async def test_a_failure_on_the_second_unit_persists_no_content(
    session_factory, uow_factory, monkeypatch
):
    """One unit failing must not leave its sibling half-written.

    The publish walks units in two passes, so a failure on the second unit happens
    after the first already inserted rows. Those rows must disappear; only the
    error status, which is written on its own session, may survive.
    """
    real_sync = SqlSenseRepo.sync
    calls = {"count": 0}

    async def fail_on_second(self, word_id, senses, cefr_map):  # noqa: ANN001
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("boom on the second unit")
        return await real_sync(self, word_id, senses, cefr_map)

    monkeypatch.setattr(SqlSenseRepo, "sync", fail_on_second)

    with pytest.raises(RuntimeError, match="boom on the second unit"):
        await GenerationWriter(uow_factory).publish(_two_unit_result())

    assert await _count(session_factory, Sense) == 0
    async with session_scope(session_factory) as session:
        statuses = set((await session.execute(select(Word.status))).scalars().all())
    # Error recording ran on an independent session, so the words survive as errors
    # rather than as half-written content.
    assert statuses == {"error"}


# --- the claim is committed before provider work ----------------------------


async def test_claim_is_durable_before_any_provider_call(session_factory, uow_factory):
    """The fence only works if the epoch is committed, not merely flushed.

    Reading it back through a separate session is the test: that is exactly what a
    competing worker in another process does.
    """
    fence = await GenerationWriter(uow_factory).claim("fenced")

    async with session_scope(session_factory) as session:
        word = (
            await session.execute(select(Word).where(Word.match_key == match_key("fenced")))
        ).scalar_one()

    assert word.generation_epoch == fence.epoch
    assert word.status == "pending"


async def test_a_second_claim_supersedes_the_first(session_factory, uow_factory):
    writer = GenerationWriter(uow_factory)
    first = await writer.claim("contested")
    second = await writer.claim("contested")

    assert second.epoch > first.epoch
    async with uow_factory() as uow:
        assert await uow.words.fence_is_current(second) is True
        assert await uow.words.fence_is_current(first) is False


# --- the independent-session escape hatch -----------------------------------


async def test_new_session_survives_a_rolled_back_unit_of_work(session_factory, uow_factory):
    """A rolled-back session cannot write, which is why the hatch exists."""
    uow = uow_factory()
    async with uow:
        await uow.words.get_or_create("doomed")
        await uow.rollback()

    # The independent session writes even though the unit of work rolled back.
    independent = SqlAlchemyUnitOfWork(session_factory)
    async with independent:
        await independent.words.get_or_create("recorded-after-rollback")
        await independent.commit()

    assert await _count(session_factory, Word) == 1


async def test_using_the_unit_of_work_outside_its_context_is_refused(session_factory):
    uow = SqlAlchemyUnitOfWork(session_factory)

    with pytest.raises(RuntimeError, match="not active"):
        _ = uow.session
