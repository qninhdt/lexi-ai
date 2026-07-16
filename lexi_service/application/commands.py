"""Transport-neutral commands for service submission and worker execution."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from lexi_ai.read_models import SearchResult
from lexi_service.identity import Principal


@dataclass(frozen=True)
class RequestContext:
    """Trusted request metadata created by an adapter after mTLS verification."""

    request_id: str
    principal: Principal | None
    deadline: datetime | None = None


@dataclass(frozen=True)
class JobReference:
    job_id: str
    status: str = "queued"
    deduplicated: bool = False


@dataclass(frozen=True)
class JobSubmission:
    operation: str
    request_id: str
    owner: Principal
    idempotency_key: str
    payload_version: int
    reference_dataset_fingerprint: str
    accepted_at: datetime
    payload: Mapping[str, str | int | None]
    maximum_age_seconds: int
    max_retries: int


@dataclass(frozen=True)
class SubmitGenerate:
    context: RequestContext
    target: SearchResult
    idempotency_key: str
    reference_dataset_fingerprint: str
    payload_version: int = 1


@dataclass(frozen=True)
class ExecuteGenerate:
    job: JobSubmission
    target: SearchResult
    generation_epoch: int
    attempt: int


@dataclass(frozen=True)
class SubmitTranslation:
    context: RequestContext
    source_kind: str
    source_id: int
    language: str
    source_hash: str
    idempotency_key: str
    reference_dataset_fingerprint: str
    payload_version: int = 1


@dataclass(frozen=True)
class ExecuteTranslation:
    job: JobSubmission
    source_kind: str
    source_id: int
    language: str
    source_hash: str
    attempt: int
