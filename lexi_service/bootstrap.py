"""Concrete composition root shared by API and worker processes."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from lexi_ai.api import Lexicon
from lexi_ai.config import Settings
from lexi_service.application.policies import ServicePolicy
from lexi_service.coordination.limits import RedisProviderGate
from lexi_service.jobs.repository import SqlJobRepository
from lexi_service.runtime import ServiceRuntime
from lexi_service.settings import ServiceSettings


class UtcClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class FileDataset:
    """Immutable reference artifact with a startup/readiness fingerprint check."""

    def __init__(self, path: str, fingerprint: str):
        self._path = Path(path)
        self.fingerprint = fingerprint

    async def ready(self) -> bool:
        if not self._path.is_file():
            return False
        digest = sha256()
        try:
            with self._path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return False
        return digest.hexdigest() == self.fingerprint


class LexiconSourcePrecondition:
    """Fence delayed translation jobs against the library's current source text."""

    def __init__(self, lexicon: Lexicon):
        self._lexicon = lexicon

    async def matches(self, source_kind: str, source_id: int, expected_hash: str) -> bool:
        return await self._lexicon.source_hash(source_kind, source_id) == expected_hash


class ServiceResources:
    def __init__(self, engine: AsyncEngine, lexicon: Lexicon):
        self._engine, self._lexicon = engine, lexicon

    async def close(self) -> None:
        await self._engine.dispose()
        library_engine = getattr(self._lexicon, "_engine", None)
        if library_engine is not None:
            await library_engine.dispose()


async def database_ready(engine: AsyncEngine) -> bool:
    """Check the service database without exposing database errors to callers."""
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    except Exception:
        return False
    return True


def policy_from_settings(settings: ServiceSettings) -> ServicePolicy:
    from datetime import timedelta

    return ServicePolicy(
        settings.max_request_bytes,
        settings.max_query_chars,
        settings.max_idempotency_key_chars,
        settings.max_page_size,
        settings.max_batch_size,
        settings.max_provider_concurrency,
        timedelta(seconds=settings.maximum_job_age_seconds),
        timedelta(seconds=settings.provider_attempt_timeout_seconds),
        settings.max_retries,
        settings.max_provider_concurrency_per_tenant,
        settings.max_outstanding_jobs_per_owner,
    )


def compose_runtime(
    service: ServiceSettings, library: Settings, *, provider_gate: RedisProviderGate | None = None
) -> ServiceRuntime:
    """Compose without calling library `init_models`; service schema is Alembic-owned."""
    service.validate_startup()
    service.validate_library_provider_settings(library)
    engine = create_async_engine(service.database_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    # Service and library effects must target the same generated-dictionary DB.
    # `LEXI_SERVICE_DATABASE_URL` is the authoritative value in service mode.
    lexicon = Lexicon.from_settings(library.model_copy(update={"db_url": service.database_url}))
    jobs = SqlJobRepository(sessions, service.max_outstanding_jobs_per_owner)
    return ServiceRuntime.compose(
        lexicon=lexicon,
        jobs=jobs,
        publisher=jobs,
        dataset=FileDataset(library.cambridge_db_path, service.reference_dataset_fingerprint),
        policy=policy_from_settings(service),
        clock=UtcClock(),
        resources=ServiceResources(engine, lexicon),
        source_preconditions=LexiconSourcePrecondition(lexicon),
        provider_gate=provider_gate,
    )
