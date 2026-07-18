# Service operations runbook

Service job schemas are Alembic-only. Never call library `init_models()` from an
API or worker process. The generated-dictionary schema must be provisioned by
the library's controlled bootstrap before the first service migration; the
service migration then adds its `generation_epoch` fencing column and its own
job/outbox tables. Deploy with expand/backfill/switch/contract migrations and
keep N/N-1 API and workers able to read queued payload version 1 until it drains.

## Compatibility matrix

| Surface | v1 compatibility rule | Rollback rule |
| --- | --- | --- |
| gRPC package | `lexi.v1`; add fields only | retain reserved field numbers |
| HTTP paths | additive endpoints/fields only | retain v1 response keys |
| job payload | version 1 handlers remain deployed until queues drain | dead-letter unsupported future versions |
| service schema | expand, backfill, switch, contract | only revert while old columns remain |
| reference data | all ready workers use one fingerprint | stop dispatch on mismatch |

Before a rolling deploy, migrate compatible schema, deploy compatible workers,
then API replicas. Reverse that order for rollback after stopping dispatch.

All API and worker replicas mount the same immutable, read-only reference
dataset at `/reference/cambridge.db`. Publish its digest as
`LEXI_SERVICE_REFERENCE_DATASET_FINGERPRINT`; a queued job carrying a different
fingerprint is rejected before provider work, and a worker refuses to start when
its mounted artifact does not hash to that fingerprint. Do not place the dataset
or service credentials in a ConfigMap. Supply database/Redis URLs and workload
certificates through the deployment's secret integration.

The Kubernetes deployments expect a secret named `lexi-service-config` with
`LEXI_SERVICE_DATABASE_URL`, `LEXI_SERVICE_REDIS_URL`,
`LEXI_SERVICE_REFERENCE_DATASET_FINGERPRINT`, all request/limit fields from
`ServiceSettings`, and any `LEXI_*` library provider settings. The manifests do
not define that secret so credentials cannot enter source control.

Provider concurrency is enforced through Redis slots across all API/worker/gRPC
replicas: `LEXI_SERVICE_MAX_PROVIDER_CONCURRENCY` is global and
`LEXI_SERVICE_MAX_PROVIDER_CONCURRENCY_PER_TENANT` is per authenticated owner.
Slots have a bounded TTL derived from the provider attempt timeout, so a crashed
process cannot retain capacity indefinitely.

When `LEXI_LLM_API_KEY` is set, service startup refuses a non-HTTPS
`LEXI_LLM_BASE_URL` except loopback endpoints used for local testing. TTS and
binary service endpoints are not part of v1; future binary storage must use a
shared durable object store, authorized delivery, and post-commit garbage
collection after the rollback retention window.

For rollback: stop outbox dispatch, retain queued jobs and job effects, restore a
compatible API/worker image, then resume dispatch. Do not delete PostgreSQL job
or effect rows. Redis may be rebuilt because PostgreSQL outbox replay restores
delivery. Workers reject mismatched dataset fingerprints and stale lease epochs.
The `generation_epoch` dictionary column is expand-only and remains present when
rolling back service migrations, so a newer library process remains compatible.

Each worker writes a durable job effect before marking the leased job successful.
If it crashes in that interval, a replay adopts the effect and completes the job
without repeating provider work. Keep job-effect rows for at least the maximum
queue retention window; they are part of the paid-provider replay guarantee.
Submission persists the client request ID with the job; structured logs therefore
carry `request_id`, `job_id`, operation, and attempt from API acceptance through
worker provider execution and effect adoption/completion.

## Integration verification

Run the hermetic suite with `uv run pytest`. For real backend tiers, point tests
at disposable local instances; the Postgres fixture creates/drops dictionary
tables and the Redis test deletes its unique stream:

```bash
LEXI_TEST_PG_URL=postgresql+asyncpg://lexi:password@localhost/lexi_test \
  uv run pytest tests/test_postgres_integration.py
LEXI_TEST_REDIS_URL=redis://localhost:6379/0 \
  uv run pytest tests/test_redis_stream_integration.py
```

## Local Compose

Compose starts the reference bootstrap, generated-dictionary bootstrap and
Alembic migration gates before the HTTP compatibility API and worker. The
default artifact is the project's synthetic/open-source-derived reference
dataset, downloaded from the public `reference-data-v1` release, cached in the
named volume, and mounted read-only at `/reference/cambridge.db`. It is not
Cambridge Dictionary data. Set both URL and SHA-256 only when pinning an
approved replacement artifact; always supply the internal token:

```bash
export LEXI_SERVICE_INTERNAL_SERVICE_TOKEN="$(openssl rand -hex 32)"
docker compose -f deploy/docker-compose.yml up --build
```

To use a different artifact, set `LEXI_REFERENCE_DATASET_URL` and
`LEXI_REFERENCE_DATASET_SHA256` together before starting Compose.

The Lexi API has no host port. Pycil joins the pre-created external
`pycil-backend` network and sends the internal token only over that network.
Supply `LEXI_LLM_*` only when exercising provider-backed jobs.
