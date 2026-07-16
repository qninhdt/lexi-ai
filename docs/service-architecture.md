# Service architecture

The library remains the dictionary engine. `lexi_service` owns transports,
durable service jobs, outbox dispatch, workers, coordination, and deployment.
PostgreSQL is authoritative; Redis Streams is at-least-once delivery only.
Fencing epochs and durable effects protect against duplicate delivery and stale
workers. See `docs/service-integration.md` and `docs/operations-runbook.md`.
