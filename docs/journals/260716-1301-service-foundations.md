---
title: "Service Foundations"
created: 2026-07-16
phase: 1
status: completed
---

# Service Foundations

## Context

Phase 1 completed the documentation-only foundation for exposing the existing
Lexi engine as a service while preserving the in-process library.

## What happened

- Established one core `Lexicon` and repository implementation: the service
  delegates to it rather than creating a second dictionary.
- Made generation and translation job-first; reads and cache hits remain
  synchronous, while submission and worker execution are distinct operations.
- Defined ownership from verified mTLS workload identity. Shared dictionary
  content stays global; jobs, idempotency records, quotas, and private results
  are owned by the derived principal or tenant.
- Defined a fenced lifecycle: conditional job transitions require the current
  attempt token and generation epoch, with replay adoption and stale attempts
  unable to commit effects.
- Reserved service schema ownership for Alembic and retained `Lexicon.init()`
  only for library-mode bootstrap; incompatible database modes fail closed.

## Decisions

Bounded request and execution controls are required before costly endpoints,
but tenant quotas and per-principal or global cost budgets are deliberately
deferred until a concrete operational need establishes the right policy.

## Next

Phase 2 is next: implement the transport-neutral service seams and isolate
service-only dependencies. No runtime code was added in Phase 1. External
publishing was not attempted.
