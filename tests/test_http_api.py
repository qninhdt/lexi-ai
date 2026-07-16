import asyncio
from types import SimpleNamespace

import httpx
import pytest

from lexi_ai.read_models import SearchResult
from lexi_service.application.commands import JobReference
from lexi_service.application.errors import ErrorCode, public_error
from lexi_service.identity import Principal
from lexi_service.observability.metrics import ServiceMetrics
from lexi_service.transport.health import HealthChecks
from lexi_service.transport.http_app import create_http_app


class GuardedQueries:
    def __init__(self):
        self.calls = []

    async def search(self, query):
        self.calls.append(query)
        if query.context.principal is None:
            raise public_error(
                ErrorCode.UNAUTHENTICATED, "Authenticated service identity is required."
            )
        return [SearchResult(query.query, None, cambridge_id=7)]


class GuardedSubmissions:
    def __init__(self):
        self.calls = []

    async def submit_generate(self, command):
        self.calls.append(command)
        if command.context.principal is None:
            raise public_error(
                ErrorCode.UNAUTHENTICATED, "Authenticated service identity is required."
            )
        return JobReference("job-42", deduplicated=True)


def runtime(queries=None, submissions=None, policy=None):
    return SimpleNamespace(
        queries=queries or GuardedQueries(),
        submissions=submissions or GuardedSubmissions(),
        policy=policy,
    )


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def add_verified_principal(app):
    @app.middleware("http")
    async def verified_principal(request, call_next):
        request.scope["lexi.verified_principal"] = Principal("service-a", "tenant-a")
        return await call_next(request)


@pytest.mark.asyncio
async def test_liveness_is_public_but_readiness_and_api_routes_default_to_denied():
    queries = GuardedQueries()
    app = create_http_app(runtime(queries=queries), HealthChecks(lambda: _ready()))

    async with client(app) as api:
        health = await api.get("/healthz")
        ready = await api.get("/readyz")
        search = await api.get("/v1/search", params={"query": "cat"})

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert ready.status_code == 401
    assert search.status_code == 401
    assert search.json()["error"]["code"] == "unauthenticated"
    assert queries.calls == []


@pytest.mark.asyncio
async def test_application_errors_are_safe_structured_json():
    class FailingQueries:
        async def search(self, _):
            raise public_error(ErrorCode.INTERNAL, "The service could not complete the request.")

    app = create_http_app(runtime(queries=FailingQueries()), HealthChecks(lambda: _ready()))
    add_verified_principal(app)

    async with client(app) as api:
        response = await api.get("/v1/search", params={"query": "cat"})

    assert response.status_code == 500
    body = response.json()["error"]
    assert body["code"] == "internal"
    assert body["message"] == "The service could not complete the request."
    assert body["incident_id"]


@pytest.mark.asyncio
async def test_generate_returns_accepted_job_and_propagates_request_id():
    submissions = GuardedSubmissions()
    app = create_http_app(runtime(submissions=submissions), HealthChecks(lambda: _ready()))

    add_verified_principal(app)

    async with client(app) as api:
        response = await api.post(
            "/v1/generations",
            headers={"x-request-id": "request-123", "Idempotency-Key": "idem-123"},
            json={
                "target": {"display": "cat", "cambridge_id": 7},
                "reference_dataset_fingerprint": "dataset-v1",
            },
        )

    assert response.status_code == 202
    assert response.json() == {
        "request_id": "request-123",
        "job_id": "job-42",
        "status": "queued",
        "deduplicated": True,
    }
    assert submissions.calls[0].context.request_id == "request-123"
    assert submissions.calls[0].context.principal == Principal("service-a", "tenant-a")


@pytest.mark.asyncio
async def test_http_deadline_bounds_synchronous_work():
    class SlowQueries:
        async def search(self, _):
            await asyncio.sleep(0.03)
            return []

    app = create_http_app(runtime(queries=SlowQueries()), HealthChecks(lambda: _ready()))
    add_verified_principal(app)

    async with client(app) as api:
        response = await api.get(
            "/v1/search", params={"query": "cat"}, headers={"x-request-deadline-ms": "1"}
        )

    assert response.status_code == 408
    assert response.json()["error"]["code"] == "deadline_exceeded"


@pytest.mark.asyncio
async def test_http_streaming_body_is_limited_without_content_length():
    app = create_http_app(
        runtime(policy=SimpleNamespace(max_request_bytes=10)), HealthChecks(lambda: _ready())
    )
    add_verified_principal(app)

    async def oversized_stream():
        yield b"{" + b"x" * 11 + b"}"

    async with client(app) as api:
        response = await api.post("/v1/generations", content=oversized_stream())

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "validation_failed"


@pytest.mark.asyncio
async def test_metrics_require_identity_and_expose_only_aggregate_counters():
    metrics = ServiceMetrics()
    app = create_http_app(runtime(), HealthChecks(lambda: _ready()), metrics=metrics)

    async with client(app) as api:
        denied = await api.get("/metrics")
    assert denied.status_code == 401

    authenticated = create_http_app(runtime(), HealthChecks(lambda: _ready()), metrics=metrics)
    add_verified_principal(authenticated)
    async with client(authenticated) as api:
        response = await api.get("/metrics")
    assert response.status_code == 200
    assert "lexi_http_requests_total" in response.text
    assert "tenant-a" not in response.text


async def _ready():
    return True
