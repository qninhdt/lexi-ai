# Service integration

`lexi.v1` is the canonical gRPC contract; HTTP is a compatibility adapter. All
non-liveness calls require mTLS workload identity. Send `x-request-id` for
correlation and `Idempotency-Key` for generation/translation submission. A
successful submission returns `202` and a job ID; poll `GET /v1/jobs/{job_id}`.

Jobs use server-owned maximum age, provider timeout, and retry cap. Clients must
not supply expiry or cancellation. Safe error bodies contain a stable code and
incident ID, never provider/database diagnostics. v1 excludes TTS and binary
storage.

## Running a service process

Install the optional dependencies with `uv sync --extra service`. Set
`LEXI_SERVICE_DATABASE_URL` to a `postgresql+asyncpg://` URL,
`LEXI_SERVICE_REDIS_URL`, and the service limits/fingerprint settings. Run
`alembic upgrade head` before starting API or worker processes. The API factory
is `uvicorn --factory lexi_service.server:create_app`; the worker is
`python -m lexi_service.worker.runner`. The worker dispatches committed outbox
records to Redis Streams before consuming them, so publishing never depends on a
Redis write inside the HTTP request transaction.

The canonical gRPC process is `python -m lexi_service.grpc_runner`. It requires
`LEXI_SERVICE_GRPC_PRIVATE_KEY_FILE`,
`LEXI_SERVICE_GRPC_CERTIFICATE_CHAIN_FILE`, and
`LEXI_SERVICE_GRPC_CLIENT_CA_FILE`; it refuses to bind without all three and
always configures gRPC client-certificate authentication.

`GET /healthz` is public liveness. `/readyz` and `/metrics` require the same
verified workload identity as v1 endpoints; metrics contain only aggregate
process counters and never tenant, prompt, payload, or credential labels.

## HTTP submission and polling

The client sends the SHA-256 fingerprint of the immutable reference dataset
mounted by the service. It owns the idempotency key and may retry the same
request/key pair; changing the payload while reusing the key returns `409`.

```bash
curl --cert client.crt --key client.key --cacert service-ca.crt \
  -H 'Idempotency-Key: generate-cat-001' \
  -H 'x-request-id: example-request-001' \
  -H 'content-type: application/json' \
  --data '{
    "target": {"display": "cat", "cambridge_id": 7},
    "reference_dataset_fingerprint": "<sha256-of-cambridge.db>",
    "payload_version": 1
  }' \
  https://lexi.example.internal/v1/generations
```

Poll `GET /v1/jobs/{job_id}` with the same workload identity. Jobs are owned by
the authenticated subject/tenant; another identity receives `404`, not job
metadata. Clients do not send cancellation, job expiry, retry caps, or provider
deadlines: those are server-owned policy values.

The service applies a server-configured outstanding-job cap per authenticated
owner and global/per-tenant provider concurrency caps. A full owner quota returns
`409 conflict`; retrying an already accepted idempotency key remains safe and
returns its original job without consuming another slot.
