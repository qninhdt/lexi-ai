"""Service-only configuration. Library settings remain in `lexi_ai.config`."""

import ipaddress
from urllib.parse import urlparse

from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LEXI_SERVICE_", extra="ignore")

    database_url: str
    redis_url: str
    reference_dataset_fingerprint: str
    max_request_bytes: int
    max_query_chars: int
    max_idempotency_key_chars: int
    max_page_size: int
    max_batch_size: int
    max_provider_concurrency: int
    max_provider_concurrency_per_tenant: int = 2
    maximum_job_age_seconds: int
    provider_attempt_timeout_seconds: int
    max_retries: int
    max_outstanding_jobs_per_owner: int = 100

    def validate_startup(self) -> None:
        if not self.database_url.startswith("postgresql+asyncpg://"):
            raise ValueError("service mode requires a PostgreSQL asyncpg database URL")
        if not self.redis_url.startswith(("redis://", "rediss://", "unix://")):
            raise ValueError("service mode requires a Redis URL")
        if not self.reference_dataset_fingerprint:
            raise ValueError("service mode requires a reference dataset fingerprint")
        if (
            min(
                self.max_request_bytes,
                self.max_query_chars,
                self.max_idempotency_key_chars,
                self.max_provider_concurrency,
                self.max_provider_concurrency_per_tenant,
                self.maximum_job_age_seconds,
                self.provider_attempt_timeout_seconds,
                self.max_outstanding_jobs_per_owner,
            )
            <= 0
        ):
            raise ValueError("service limits and timeouts must be positive")

    def validate_library_provider_settings(self, library) -> None:
        """Do not start a service that would send an API key over plaintext."""
        api_key = getattr(library, "llm_api_key", "")
        if not api_key:
            return
        base_url = getattr(library, "llm_base_url", "")
        parsed = urlparse(base_url)
        if parsed.scheme == "https":
            return
        host = parsed.hostname
        if parsed.scheme == "http" and host and _is_loopback(host):
            return
        raise ValueError("service mode requires an HTTPS LLM URL when an API key is configured")


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
