"""mTLS gRPC process entrypoint for the canonical `lexi.v1` transport."""

import asyncio
import os
from pathlib import Path

from redis.asyncio import Redis

from lexi_ai.config import get_settings
from lexi_service.bootstrap import compose_runtime, database_ready
from lexi_service.coordination.limits import RedisProviderGate
from lexi_service.settings import ServiceSettings
from lexi_service.transport.grpc_server import create_grpc_server
from lexi_service.transport.health import dependencies_ready


def _read_secret(name: str) -> bytes:
    path = os.environ.get(name)
    if not path:
        raise ValueError(f"{name} must point to an mTLS credential file")
    return Path(path).read_bytes()


async def run() -> None:
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
    sessions = runtime.submissions._publisher.sessions
    engine = sessions.kw["bind"]

    async def redis_ready() -> bool:
        try:
            await redis.ping()
        except Exception:
            return False
        return True

    dataset = runtime.dataset
    health = dependencies_ready(
        lambda: database_ready(engine),
        redis_ready,
        dataset.ready if dataset is not None and hasattr(dataset, "ready") else _always_ready,
    )
    server = create_grpc_server(
        runtime,
        health,
        private_key=_read_secret("LEXI_SERVICE_GRPC_PRIVATE_KEY_FILE"),
        certificate_chain=_read_secret("LEXI_SERVICE_GRPC_CERTIFICATE_CHAIN_FILE"),
        client_ca=_read_secret("LEXI_SERVICE_GRPC_CLIENT_CA_FILE"),
    )
    address = os.environ.get("LEXI_SERVICE_GRPC_BIND", "0.0.0.0:9443")
    if not server.add_mtls_port(address):
        raise RuntimeError(f"could not bind gRPC server to {address}")
    try:
        await server.start()
        await server.wait_for_termination()
    finally:
        await server.stop(5)
        await redis.aclose()
        await runtime.close()


async def _always_ready() -> bool:
    return True


if __name__ == "__main__":
    asyncio.run(run())
