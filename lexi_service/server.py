"""ASGI factory for `uvicorn --factory lexi_service.server:create_app`."""

from redis.asyncio import Redis

from lexi_ai.config import get_settings
from lexi_service.bootstrap import compose_runtime, database_ready
from lexi_service.coordination.limits import RedisProviderGate
from lexi_service.settings import ServiceSettings
from lexi_service.transport.health import HealthChecks
from lexi_service.transport.http_app import create_http_app


def create_app():
    settings = ServiceSettings()
    redis = Redis.from_url(settings.redis_url)
    runtime = compose_runtime(
        settings,
        get_settings(),
        provider_gate=RedisProviderGate(
            redis,
            settings.max_provider_concurrency,
            settings.max_provider_concurrency_per_tenant,
            settings.provider_attempt_timeout_seconds * 2_000 + 1_000,
        ),
    )

    async def ready() -> bool:
        if runtime._closed:
            return False
        engine = getattr(runtime.submissions._publisher, "sessions", None)
        if engine is None:
            return False
        bind = engine.kw.get("bind")
        if bind is None or not await database_ready(bind):
            return False
        dataset = runtime.dataset
        if dataset is not None and hasattr(dataset, "ready") and not await dataset.ready():
            return False
        try:
            await redis.ping()
        except Exception:
            return False
        return True

    app = create_http_app(
        runtime,
        HealthChecks(ready),
        internal_service_token=settings.internal_service_token,
        internal_service_subject=settings.internal_service_subject,
    )

    @app.on_event("shutdown")
    async def close_runtime() -> None:
        try:
            await redis.aclose()
        finally:
            await runtime.close()

    return app
