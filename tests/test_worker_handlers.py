from dataclasses import dataclass

import pytest

from lexi_service.application.errors import ApplicationError
from lexi_service.jobs.outbox import OutboxEnvelope
from lexi_service.worker.handlers import HandlerGate


@dataclass
class Job:
    operation: str = "translate"
    reference_dataset_fingerprint: str = "dataset"
    payload: dict = None


class Loader:
    async def load(self, _):
        return Job(payload={"source_kind": "sense", "source_id": 1, "source_hash": "old"})


class Sources:
    async def matches(self, *_):
        return False


async def test_worker_rejects_source_changed_before_provider_execution():
    gate = HandlerGate(Loader(), "dataset", Sources())
    with pytest.raises(ApplicationError) as raised:
        await gate.load_verified(OutboxEnvelope("event", "job", "translate", 1))
    assert raised.value.error.code.value == "precondition_failed"
