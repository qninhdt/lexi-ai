import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from lexi_service.jobs.effects import JobEffects
from lexi_service.jobs.models import ServiceBase


@pytest.fixture
async def effects():
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as connection:
        await connection.run_sync(ServiceBase.metadata.create_all)
    try:
        yield JobEffects(async_sessionmaker(engine, expire_on_commit=False))
    finally:
        await engine.dispose()


async def test_replay_adopts_committed_effect_instead_of_overwriting(effects):
    assert await effects.adopt_or_record("job", "generation", {"word_id": 7}) == {"word_id": 7}
    assert await effects.adopt_or_record("job", "generation", {"word_id": 8}) == {"word_id": 7}
    assert await effects.get("job", "generation") == {"word_id": 7}
