"""Async gRPC adapter for the canonical `lexi.v1` contract."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import grpc

from lexi_ai.read_models import SearchResult
from lexi_service.application.commands import RequestContext, SubmitGenerate, SubmitTranslation
from lexi_service.application.errors import ApplicationError
from lexi_service.application.queries import GetJobQuery, LookupEntryQuery, SearchQuery, StatsQuery
from lexi_service.proto.lexi.v1 import lexi_pb2, lexi_pb2_grpc
from lexi_service.runtime import ServiceRuntime
from lexi_service.security.auth import principal_from_grpc_auth_context
from lexi_service.transport.health import HealthChecks
from lexi_service.transport.mapping import entry, search_target

_STATUS = {
    "unauthenticated": grpc.StatusCode.UNAUTHENTICATED,
    "forbidden": grpc.StatusCode.PERMISSION_DENIED,
    "validation_failed": grpc.StatusCode.INVALID_ARGUMENT,
    "not_found": grpc.StatusCode.NOT_FOUND,
    "conflict": grpc.StatusCode.ALREADY_EXISTS,
    "deadline_exceeded": grpc.StatusCode.DEADLINE_EXCEEDED,
    "precondition_failed": grpc.StatusCode.FAILED_PRECONDITION,
    "internal": grpc.StatusCode.INTERNAL,
}


class LexiGrpcServicer(lexi_pb2_grpc.LexiServiceServicer):
    def __init__(self, runtime: ServiceRuntime, health: HealthChecks | None = None):
        self._runtime = runtime
        self._health = health

    def _context(self, rpc_context):
        if cached := getattr(rpc_context, "_lexi_request_context", None):
            return cached
        principal = principal_from_grpc_auth_context(rpc_context.auth_context())
        request_id = dict(rpc_context.invocation_metadata()).get("x-request-id", uuid4().hex)
        remaining = getattr(rpc_context, "time_remaining", lambda: None)()
        deadline = (
            None if remaining is None else datetime.now(timezone.utc) + timedelta(seconds=remaining)
        )
        value = RequestContext(request_id, principal, deadline)
        rpc_context._lexi_request_context = value
        return value

    async def _call(self, rpc_context, operation):
        try:
            return await operation()
        except ApplicationError as exc:
            await rpc_context.abort(_STATUS[exc.error.code.value], exc.error.message)

    async def Search(self, request, context):
        result = await self._call(
            context,
            lambda: self._runtime.queries.search(
                SearchQuery(self._context(context), request.query)
            ),
        )
        return lexi_pb2.SearchResponse(
            meta=lexi_pb2.ResponseMeta(request_id=self._context(context).request_id),
            results=[search_target(x) for x in result],
        )

    async def Lookup(self, request, context):
        result = await self._call(
            context,
            lambda: self._runtime.queries.lookup(
                LookupEntryQuery(self._context(context), request.word_id)
            ),
        )
        return lexi_pb2.EntryResponse(
            meta=lexi_pb2.ResponseMeta(request_id=self._context(context).request_id),
            entry=entry(result),
        )

    async def SubmitGenerate(self, request, context):
        meta = dict(context.invocation_metadata())
        target = SearchResult(
            request.target.display,
            request.target.entry_type or None,
            request.target.score,
            request.target.lexi_word_id or None,
            request.target.cambridge_id or None,
            request.target.gloss or None,
        )
        result = await self._call(
            context,
            lambda: self._runtime.submissions.submit_generate(
                SubmitGenerate(
                    self._context(context),
                    target,
                    meta.get("idempotency-key", ""),
                    request.reference_dataset_fingerprint,
                    request.payload_version or 1,
                )
            ),
        )
        return lexi_pb2.JobResponse(
            meta=lexi_pb2.ResponseMeta(request_id=self._context(context).request_id),
            job=lexi_pb2.Job(
                job_id=result.job_id,
                status=result.status,
                deduplicated=result.deduplicated,
                operation="generate",
            ),
        )

    async def GetJob(self, request, context):
        result = await self._call(
            context,
            lambda: self._runtime.queries.get_job(
                GetJobQuery(self._context(context), request.job_id)
            ),
        )
        return lexi_pb2.JobResponse(
            meta=lexi_pb2.ResponseMeta(request_id=self._context(context).request_id),
            job=lexi_pb2.Job(
                job_id=result.reference.job_id,
                status=result.reference.status,
                deduplicated=result.reference.deduplicated,
                operation=result.operation,
            ),
        )

    async def SubmitTranslation(self, request, context):
        meta = dict(context.invocation_metadata())
        result = await self._call(
            context,
            lambda: self._runtime.submissions.submit_translation(
                SubmitTranslation(
                    self._context(context),
                    request.source_kind,
                    request.source_id,
                    request.language,
                    request.source_hash,
                    meta.get("idempotency-key", ""),
                    request.reference_dataset_fingerprint,
                    request.payload_version or 1,
                )
            ),
        )
        return lexi_pb2.JobResponse(
            meta=lexi_pb2.ResponseMeta(request_id=self._context(context).request_id),
            job=lexi_pb2.Job(
                job_id=result.job_id,
                status=result.status,
                deduplicated=result.deduplicated,
                operation="translate",
            ),
        )

    async def Stats(self, request, context):
        result = await self._call(
            context, lambda: self._runtime.queries.stats(StatsQuery(self._context(context)))
        )
        return lexi_pb2.StatsResponse(
            meta=lexi_pb2.ResponseMeta(request_id=self._context(context).request_id),
            words_by_status=result.words_by_status,
            senses=result.senses,
            examples=result.examples,
            tags=result.tags,
        )

    async def Health(self, request, context):
        return lexi_pb2.HealthResponse(status="ok")

    async def Readiness(self, request, context):
        if self._context(context).principal is None:
            await context.abort(
                grpc.StatusCode.UNAUTHENTICATED, "Authenticated service identity is required."
            )
        ready = True if self._health is None else await self._health.readiness()
        if not ready:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "Service is not ready.")
        return lexi_pb2.HealthResponse(status="ready")


class MtlsGrpcServer:
    """Own an async gRPC server that exposes only secure client-certificate binding."""

    def __init__(self, server: grpc.aio.Server, credentials: grpc.ServerCredentials):
        self._server = server
        self._credentials = credentials

    def add_mtls_port(self, address: str) -> int:
        return self._server.add_secure_port(address, self._credentials)

    async def start(self) -> None:
        await self._server.start()

    async def stop(self, grace: float | None = None) -> None:
        await self._server.stop(grace)

    async def wait_for_termination(self) -> None:
        await self._server.wait_for_termination()


def create_grpc_server(
    runtime: ServiceRuntime,
    health: HealthChecks,
    *,
    private_key: bytes,
    certificate_chain: bytes,
    client_ca: bytes,
) -> MtlsGrpcServer:
    """Create an mTLS-only async server; its raw server is never exposed."""
    if not all((private_key, certificate_chain, client_ca)):
        raise ValueError("gRPC mTLS requires a private key, certificate chain, and client CA.")
    options = [
        ("grpc.max_receive_message_length", runtime.policy.max_request_bytes),
        ("grpc.max_send_message_length", runtime.policy.max_request_bytes),
    ]
    server = grpc.aio.server(options=options)
    lexi_pb2_grpc.add_LexiServiceServicer_to_server(LexiGrpcServicer(runtime, health), server)
    credentials = grpc.ssl_server_credentials(
        ((private_key, certificate_chain),),
        root_certificates=client_ca,
        require_client_auth=True,
    )
    return MtlsGrpcServer(server, credentials)
