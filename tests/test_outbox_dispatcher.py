from datetime import datetime

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from lexi_service.jobs.dispatcher import OutboxDispatcher
from lexi_service.jobs.models import OutboxEventRow, ServiceBase
from lexi_service.jobs.outbox import OutboxEnvelope, SqlOutbox


class CapturingPublisher:
    def __init__(self):
        self.messages = []

    async def publish(self, envelope):
        self.messages.append(envelope)


@pytest.fixture
async def outbox():
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(ServiceBase.metadata.create_all)
    async with sessions.begin() as session:
        session.add(
            OutboxEventRow(
                event_id="event-1",
                job_id="job-1",
                operation="generate",
                payload_version=1,
                created_at=datetime.now(),
            )
        )
    try:
        yield SqlOutbox(sessions)
    finally:
        await engine.dispose()


async def test_dispatch_marks_only_successfully_published_events(outbox):
    publisher = CapturingPublisher()
    dispatched = await OutboxDispatcher(outbox, publisher).dispatch_once()

    assert dispatched == 1
    assert publisher.messages == [OutboxEnvelope("event-1", "job-1", "generate", 1)]
    assert await outbox.pending(10) == []
