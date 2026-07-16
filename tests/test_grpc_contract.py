import pytest

from lexi_ai.read_models import SearchResult
from lexi_service.application.errors import ErrorCode, public_error
from lexi_service.proto.lexi.v1 import lexi_pb2, lexi_pb2_grpc
from lexi_service.transport.grpc_server import LexiGrpcServicer
from lexi_service.transport.health import HealthChecks


class RpcAborted(Exception):
    def __init__(self, code, details):
        self.code = code
        self.details = details


class FakeRpcContext:
    def __init__(self, *, metadata=(), authenticated=False):
        self._metadata = metadata
        self._authenticated = authenticated

    def invocation_metadata(self):
        return self._metadata

    def auth_context(self):
        if self._authenticated:
            return {"transport_security_type": [b"ssl"], "x509_common_name": [b"service-a"]}
        return {}

    async def abort(self, code, details):
        raise RpcAborted(code, details)


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


class FakeRuntime:
    def __init__(self):
        self.queries = GuardedQueries()
        self.submissions = None


def test_generated_proto_uses_versioned_package_and_exposes_service():
    assert lexi_pb2.DESCRIPTOR.package == "lexi.v1"
    assert "LexiService" in lexi_pb2.DESCRIPTOR.services_by_name
    assert hasattr(lexi_pb2_grpc, "LexiServiceStub")
    assert hasattr(lexi_pb2_grpc, "add_LexiServiceServicer_to_server")


@pytest.mark.asyncio
async def test_health_is_public_without_a_client_certificate():
    response = await LexiGrpcServicer(FakeRuntime()).Health(
        lexi_pb2.HealthRequest(), FakeRpcContext()
    )

    assert response.status == "ok"


@pytest.mark.asyncio
async def test_readiness_requires_a_client_certificate_and_reports_dependency_state():
    servicer = LexiGrpcServicer(FakeRuntime(), HealthChecks(lambda: _not_ready()))

    with pytest.raises(RpcAborted) as denied:
        await servicer.Readiness(lexi_pb2.HealthRequest(), FakeRpcContext())
    with pytest.raises(RpcAborted) as unavailable:
        await servicer.Readiness(lexi_pb2.HealthRequest(), FakeRpcContext(authenticated=True))

    assert denied.value.code.name == "UNAUTHENTICATED"
    assert unavailable.value.code.name == "UNAVAILABLE"


@pytest.mark.asyncio
async def test_non_liveness_rpc_default_denies_and_does_not_call_provider_logic():
    runtime = FakeRuntime()
    context = FakeRpcContext()

    with pytest.raises(RpcAborted) as raised:
        await LexiGrpcServicer(runtime).Search(lexi_pb2.SearchRequest(query="cat"), context)

    assert raised.value.code.name == "UNAUTHENTICATED"
    assert "Authenticated service identity" in raised.value.details
    assert runtime.queries.calls[0].context.principal is None


@pytest.mark.asyncio
async def test_grpc_preserves_client_request_id_in_response_and_command():
    runtime = FakeRuntime()
    context = FakeRpcContext(metadata=(("x-request-id", "request-123"),), authenticated=True)

    response = await LexiGrpcServicer(runtime).Search(lexi_pb2.SearchRequest(query="cat"), context)

    assert response.meta.request_id == "request-123"
    assert response.results[0].display == "cat"
    assert runtime.queries.calls[0].context.request_id == "request-123"


async def _not_ready():
    return False
