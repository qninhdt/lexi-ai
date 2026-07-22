# Service API adapters

Phase 3 introduces `lexi.v1` as the canonical service contract. gRPC and HTTP
are deliberately thin: both construct the same application commands and queries,
so persistence and provider behavior remain in the shared core.

Operationally, only liveness is public. gRPC can bind only with server
credentials that require a client certificate; HTTP requires a verified
certificate identity supplied by the ASGI TLS integration. Both preserve request
IDs, expose protected readiness, constrain request work by deadlines, and return
safe structured errors. HTTP streams are bounded before JSON parsing, while gRPC
sets equivalent message limits. Accepted generation and translation requests
return durable job references; neither transport exposes cancellation, remote
force, or client-owned job expiry.
