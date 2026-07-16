# Kubernetes deployment

Run API, worker, and Alembic migration as separate workloads. Mount immutable
reference artifacts at the same path in API and worker pods; readiness must fail
when their fingerprints differ. Terminate mTLS only at an identity-aware proxy
that injects the verified client certificate into the ASGI scope. Do not expose
TTS or binary result storage in service v1.
