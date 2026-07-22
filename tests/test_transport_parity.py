"""Contract parity for the shared HTTP and canonical gRPC submission path."""

from types import SimpleNamespace

import httpx

from lexi_service.application.commands import JobReference
from lexi_service.identity import Principal
from lexi_service.proto.lexi.v1 import lexi_pb2
from lexi_service.transport.grpc_server import LexiGrpcServicer
from lexi_service.transport.health import HealthChecks
from lexi_service.transport.http_app import create_http_app


class Submissions:
    def __init__(self):
        self.calls = []

    async def submit_generate(self, command):
        self.calls.append(command)
        return JobReference("job-42", deduplicated=True)


class RpcContext:
    def invocation_metadata(self):
        return (("x-request-id", "request-123"), ("idempotency-key", "idem-123"))

    def auth_context(self):
        return {"transport_security_type": [b"ssl"], "x509_common_name": [b"service-a"]}

    async def abort(self, code, details):  # pragma: no cover - this test is authenticated
        raise AssertionError(f"unexpected abort: {code}: {details}")


async def _ready():
    return True


async def test_http_and_grpc_submit_generate_create_the_same_application_command():
    grpc_submissions = Submissions()
    grpc_runtime = SimpleNamespace(submissions=grpc_submissions)
    grpc = LexiGrpcServicer(grpc_runtime)
    response = await grpc.SubmitGenerate(
        lexi_pb2.SubmitGenerateRequest(
            target=lexi_pb2.SearchTarget(display="cat", cambridge_id=7),
            reference_dataset_fingerprint="dataset-v1",
            payload_version=1,
            generation_strategy="function_calling",
        ),
        RpcContext(),
    )

    http_submissions = Submissions()
    http_runtime = SimpleNamespace(submissions=http_submissions, policy=None)
    app = create_http_app(http_runtime, HealthChecks(_ready))

    @app.middleware("http")
    async def verified_principal(request, call_next):
        request.scope["lexi.verified_principal"] = Principal("service-a")
        return await call_next(request)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://test"
    ) as client:
        accepted = await client.post(
            "/v1/generations",
            headers={"x-request-id": "request-123", "Idempotency-Key": "idem-123"},
            json={
                "target": {"display": "cat", "cambridge_id": 7},
                "reference_dataset_fingerprint": "dataset-v1",
                "payload_version": 1,
                "generation_strategy": "function_calling",
            },
        )

    assert response.job.job_id == "job-42"
    assert accepted.status_code == 202
    grpc_command, http_command = grpc_submissions.calls[0], http_submissions.calls[0]
    assert grpc_command.context == http_command.context
    assert grpc_command.target == http_command.target
    assert grpc_command.idempotency_key == http_command.idempotency_key
    assert grpc_command.reference_dataset_fingerprint == http_command.reference_dataset_fingerprint
    assert grpc_command.payload_version == http_command.payload_version
    assert grpc_command.generation_strategy == "function_calling"
    assert http_command.generation_strategy == "function_calling"
