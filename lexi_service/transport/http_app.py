"""FastAPI compatibility adapter over the transport-neutral application core."""

import asyncio
import hmac
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from lexi_ai.read_models import SearchResult
from lexi_service.application.commands import (
    GenerateQuestions,
    GradeQuestion,
    RequestContext,
    SubmitGenerate,
    SubmitTranslation,
)
from lexi_service.application.errors import ApplicationError, ErrorCode, public_error
from lexi_service.application.queries import (
    GetJobQuery,
    GetQuestionQuery,
    GetSensesQuery,
    ListQuestionsQuery,
    LookupEntryQuery,
    SearchQuery,
    StatsQuery,
)
from lexi_service.identity import Principal
from lexi_service.observability.metrics import ServiceMetrics
from lexi_service.runtime import ServiceRuntime
from lexi_service.security.auth import principal_from_asgi_scope
from lexi_service.transport.health import HealthChecks
from lexi_service.transport.mapping import (
    HTTP_STATUS,
    entry,
    error_body,
    json_message,
    question_presentation,
    search_target,
    sense,
)

CertificateVerifier = Callable[[bytes], Awaitable[Principal | None]]


class RequestBodyLimitMiddleware:
    """Bound request bodies without consuming the downstream ASGI stream."""

    def __init__(self, app, max_request_bytes: int):
        self.app = app
        self.max_request_bytes = max_request_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = scope.get("headers", ())
        for name, value in content_length:
            if name.lower() != b"content-length":
                continue
            try:
                if int(value) <= self.max_request_bytes:
                    break
            except ValueError:
                pass
            await self._too_large(scope, receive, send)
            return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.max_request_bytes:
                await self._too_large(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        async def replay_body():
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay_body, send)

    async def _too_large(self, scope, receive, send) -> None:
        error = public_error(ErrorCode.VALIDATION, "Request body is too large.").error
        await JSONResponse(error_body(error), status_code=413)(scope, receive, send)

class SearchTargetBody(BaseModel):
    display: str
    entry_type: str | None = None
    score: float = 0
    lexi_word_id: int | None = None
    cambridge_id: int | None = None
    gloss: str | None = None


class GenerateBody(BaseModel):
    target: SearchTargetBody
    reference_dataset_fingerprint: str
    payload_version: int = Field(1, ge=1)


class TranslationBody(BaseModel):
    source_kind: str
    source_id: int = Field(gt=0)
    language: str
    source_hash: str
    reference_dataset_fingerprint: str
    payload_version: int = Field(1, ge=1)


class QuestionGenerationBody(BaseModel):
    word_id: int = Field(gt=0)
    sense_id: int = Field(gt=0)
    formats: list[str] = Field(min_length=1, max_length=32)
    count: int = Field(1, ge=1, le=32)


class GradeQuestionBody(BaseModel):
    """Opaque answer; application code validates it against stored answer_kind."""

    answer: object


def create_http_app(
    runtime: ServiceRuntime,
    health: HealthChecks,
    *,
    certificate_verifier: CertificateVerifier | None = None,
    metrics: ServiceMetrics | None = None,
    internal_service_token: str = "",
    internal_service_subject: str = "pycil",
) -> FastAPI:
    app = FastAPI(title="Lexi Service", version="1.0.0")
    metrics = metrics or ServiceMetrics()
    policy = getattr(runtime, "policy", None)
    if policy is not None:
        app.add_middleware(RequestBodyLimitMiddleware, max_request_bytes=policy.max_request_bytes)

    def context(request: Request) -> RequestContext:
        deadline = None
        if value := request.headers.get("x-request-deadline-ms"):
            try:
                deadline = datetime.now(timezone.utc) + timedelta(milliseconds=int(value))
            except ValueError as exc:
                raise public_error(ErrorCode.VALIDATION, "Request deadline is invalid.") from exc
        return RequestContext(
            request.headers.get("x-request-id", uuid4().hex),
            principal_from_asgi_scope(request.scope),
            deadline,
        )

    @app.middleware("http")
    async def require_verified_client_certificate(request: Request, call_next):
        if request.url.path == "/healthz":
            return await call_next(request)

        principal = principal_from_asgi_scope(request.scope)
        provided_token = request.headers.get("x-lexi-service-token", "")
        if (
            principal is None
            and internal_service_token
            and hmac.compare_digest(provided_token, internal_service_token)
        ):
            principal = Principal(internal_service_subject)
            request.scope["lexi.verified_principal"] = principal
        certificate = request.scope.get("ssl_client_cert")
        has_certificate = isinstance(certificate, bytes)
        should_verify = principal is None and certificate_verifier is not None and has_certificate
        if should_verify:
            principal = await certificate_verifier(certificate)
            if principal is not None:
                request.scope["lexi.verified_principal"] = principal
        if principal is None:
            error = public_error(
                ErrorCode.UNAUTHENTICATED, "Authenticated service identity is required."
            ).error
            return JSONResponse(error_body(error), status_code=HTTP_STATUS[error.code])
        return await call_next(request)

    @app.middleware("http")
    async def enforce_deadline(request: Request, call_next):
        value = request.headers.get("x-request-deadline-ms")
        if value is None:
            return await call_next(request)
        try:
            timeout_seconds = int(value) / 1000
        except ValueError:
            error = public_error(ErrorCode.VALIDATION, "Request deadline is invalid.").error
            return JSONResponse(error_body(error), status_code=400)
        if timeout_seconds <= 0:
            error = public_error(ErrorCode.DEADLINE_EXCEEDED, "Request deadline elapsed.").error
            return JSONResponse(error_body(error), status_code=408)
        try:
            return await asyncio.wait_for(call_next(request), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            error = public_error(ErrorCode.DEADLINE_EXCEEDED, "Request deadline elapsed.").error
            return JSONResponse(error_body(error), status_code=408)

    @app.middleware("http")
    async def collect_request_metrics(request: Request, call_next):
        response = await call_next(request)
        metrics.increment("lexi_http_requests_total")
        metrics.increment(f"lexi_http_status_{response.status_code}_total")
        return response

    @app.exception_handler(ApplicationError)
    async def application_error(_, exc: ApplicationError):
        return JSONResponse(error_body(exc.error), status_code=HTTP_STATUS[exc.error.code])

    @app.get("/healthz")
    async def liveness():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readiness(request: Request):
        if context(request).principal is None:
            raise public_error(
                ErrorCode.UNAUTHENTICATED, "Authenticated service identity is required."
            )
        if not await health.readiness():
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return {"status": "ready"}

    @app.get("/metrics")
    async def prometheus_metrics():
        return PlainTextResponse(metrics.prometheus(), media_type="text/plain; version=0.0.4")

    @app.get("/v1/search")
    async def search(request: Request, query: str):
        ctx = context(request)
        values = await runtime.queries.search(SearchQuery(ctx, query))
        return {
            "request_id": ctx.request_id,
            "results": [json_message(search_target(x)) for x in values],
        }

    @app.get("/v1/entries/{word_id}")
    async def lookup(request: Request, word_id: int):
        ctx = context(request)
        value = await runtime.queries.lookup(LookupEntryQuery(ctx, word_id))
        return {"request_id": ctx.request_id, "entry": json_message(entry(value))}

    @app.get("/v1/senses")
    async def get_senses(request: Request, ids: list[int] = Query(alias="id")):  # noqa: B008
        ctx = context(request)
        values = await runtime.queries.get_senses(GetSensesQuery(ctx, ids))
        return {
            "request_id": ctx.request_id,
            "senses": [json_message(sense(value)) for value in values],
        }

    @app.get("/v1/questions/{question_id}")
    async def get_question(request: Request, question_id: int):
        ctx = context(request)
        value = await runtime.queries.get_question(GetQuestionQuery(ctx, question_id))
        return {"request_id": ctx.request_id, "question": question_presentation(value)}

    @app.get("/v1/questions")
    async def list_questions(request: Request, sense_id: int, format: str | None = None):
        ctx = context(request)
        values = await runtime.queries.list_questions(ListQuestionsQuery(ctx, sense_id, format))
        return {
            "request_id": ctx.request_id,
            "questions": [question_presentation(value) for value in values],
        }

    @app.post("/v1/questions/generations")
    async def generate_questions(request: Request, body: QuestionGenerationBody):
        ctx = context(request)
        values = await runtime.queries.generate_questions(
            GenerateQuestions(ctx, body.word_id, body.sense_id, tuple(body.formats), body.count)
        )
        return {
            "request_id": ctx.request_id,
            "questions": [question_presentation(value) for value in values],
        }

    @app.post("/v1/questions/{question_id}/grade")
    async def grade_question(request: Request, question_id: int, body: GradeQuestionBody):
        ctx = context(request)
        value = await runtime.queries.grade_question(GradeQuestion(ctx, question_id, body.answer))
        return {
            "request_id": ctx.request_id,
            "correct": value.correct,
            "score": value.score,
            "kind": value.kind,
            "feedback": value.feedback,
        }

    @app.post("/v1/generations", status_code=202)
    async def submit_generate(
        request: Request,
        body: GenerateBody,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        ctx = context(request)
        target = SearchResult(**body.target.model_dump())
        job = await runtime.submissions.submit_generate(
            SubmitGenerate(
                ctx,
                target,
                idempotency_key or "",
                body.reference_dataset_fingerprint,
                body.payload_version,
            )
        )
        return {
            "request_id": ctx.request_id,
            "job_id": job.job_id,
            "status": job.status,
            "deduplicated": job.deduplicated,
        }

    @app.post("/v1/translations", status_code=202)
    async def submit_translation(
        request: Request,
        body: TranslationBody,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        ctx = context(request)
        job = await runtime.submissions.submit_translation(
            SubmitTranslation(
                ctx,
                body.source_kind,
                body.source_id,
                body.language,
                body.source_hash,
                idempotency_key or "",
                body.reference_dataset_fingerprint,
                body.payload_version,
            )
        )
        return {
            "request_id": ctx.request_id,
            "job_id": job.job_id,
            "status": job.status,
            "deduplicated": job.deduplicated,
        }

    @app.get("/v1/jobs/{job_id}")
    async def get_job(request: Request, job_id: str):
        ctx = context(request)
        job = await runtime.queries.get_job(GetJobQuery(ctx, job_id))
        return {
            "request_id": ctx.request_id,
            "job_id": job.reference.job_id,
            "status": job.reference.status,
            "deduplicated": job.reference.deduplicated,
            "operation": job.operation,
            "result": job.result if job.reference.status == "succeeded" else None,
            "error": (
                {"code": job.error_code}
                if job.reference.status in {"failed", "expired", "superseded", "dead_letter"}
                and job.error_code is not None
                else None
            ),
        }

    @app.get("/v1/stats")
    async def stats(request: Request):
        ctx = context(request)
        value = await runtime.queries.stats(StatsQuery(ctx))
        return {
            "request_id": ctx.request_id,
            "words_by_status": value.words_by_status,
            "senses": value.senses,
            "examples": value.examples,
            "tags": value.tags,
        }

    return app
