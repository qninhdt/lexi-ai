"""Bounded-execution policy injected by service composition."""

from dataclasses import dataclass
from datetime import timedelta


@dataclass(frozen=True)
class ServicePolicy:
    max_request_bytes: int
    max_query_chars: int
    max_idempotency_key_chars: int
    max_page_size: int
    max_batch_size: int
    max_provider_concurrency: int
    maximum_job_age: timedelta
    provider_attempt_timeout: timedelta
    max_retries: int
    max_provider_concurrency_per_tenant: int = 2
    max_outstanding_jobs_per_owner: int = 100

    def __post_init__(self) -> None:
        if any(
            value <= 0
            for value in (
                self.max_query_chars,
                self.max_request_bytes,
                self.max_idempotency_key_chars,
                self.max_page_size,
                self.max_batch_size,
                self.max_provider_concurrency,
            )
        ):
            raise ValueError("service limits must be positive")
        if self.maximum_job_age <= timedelta() or self.provider_attempt_timeout <= timedelta():
            raise ValueError("service timeouts must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.max_outstanding_jobs_per_owner <= 0:
            raise ValueError("outstanding job quota must be positive")
