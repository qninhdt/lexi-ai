"""Worker dispatch gate: validate immutable dataset/source inputs before effects."""

from typing import Protocol

from lexi_service.application.errors import ErrorCode, public_error
from lexi_service.jobs.outbox import OutboxEnvelope


class WorkerJob(Protocol):
    operation: str
    reference_dataset_fingerprint: str
    payload: dict


class JobLoader(Protocol):
    async def load(self, job_id: str) -> WorkerJob | None: ...


class SourceVerifier(Protocol):
    async def matches(self, source_kind: str, source_id: int, source_hash: str) -> bool: ...


class HandlerGate:
    def __init__(
        self, loader: JobLoader, dataset_fingerprint: str, source_verifier: SourceVerifier
    ):
        self._loader, self._dataset, self._sources = loader, dataset_fingerprint, source_verifier

    async def load_verified(self, envelope: OutboxEnvelope) -> WorkerJob:
        job = await self._loader.load(envelope.job_id)
        if job is None or job.operation != envelope.operation:
            raise public_error(ErrorCode.NOT_FOUND, "Job was not found.")
        if job.reference_dataset_fingerprint != self._dataset:
            raise public_error(ErrorCode.PRECONDITION_FAILED, "Reference dataset does not match.")
        if job.operation == "translate" and not await self._sources.matches(
            job.payload["source_kind"], job.payload["source_id"], job.payload["source_hash"]
        ):
            raise public_error(ErrorCode.PRECONDITION_FAILED, "Source content changed.")
        return job
