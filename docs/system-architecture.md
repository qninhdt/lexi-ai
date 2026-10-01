# Lexi-AI architecture

`Lexicon` in `lexi_ai/api.py` composes a library, **not** an HTTP API, scheduler
or stateful learner service. Callers choose a Cambridge available handle from
`search(query, include_available=True)`, then call `generate(available_id)`.
Search, Word reads, stored Question reads and translation-cache hits do not call
models. The original Cambridge SQLite snapshot is opened read-only; only the
separate generated database is migrated or written.

```
Cambridge SQLite (read-only) → available handle → selected LLM Word output
                                                   ↓ atomic DB publication
generated database → neutral Word/Sense → Theme-specific content → Questions
                              ↘ explicit Sense Linking        ↘ grading
                                 Decision Model → official fallback adapter
```

## Storage and ownership

- `inference/` owns shared model transports, configuration and Jinja/JSON-e rendering,
  not task prompts or rubrics. Each domain owns its service, storage SQL and prompts.
  `db/` contains generic transaction, aggregation, bulk insertion, pagination and
  random-slot helpers. Import contracts forbid domain/schema dependencies from either
  shared infrastructure package. There are no repository/port/use-case layers.

- The generated DB (`schema.py`) stores neutral definitions/examples with
  `theme_id = NULL`; themed content and Questions require their **exact** Theme
  namespace. No themed read silently substitutes neutral text.
- Each Sense has one relational Definition per namespace, enforced by unique indexes
  for neutral and themed rows. Public and structured-output fields use `definition`.
  Only Question has dense `position`/`type_position`; Senses are read by ID.
- Word/Sense reads aggregate child tables at query time in one statement; there is
  no materialized Word/Sense JSON. Generation and grading use narrow projections.
  Child publication, relation-target lookup and Cambridge example reads are batched.
  Relation target evidence and Sense Linking candidates are deduplicated within the same
  statement snapshot. Pattern pages bound pattern rows, not entire Sense banks,
  and separately project one form collection per distinct Sense in the page.
  SQLite publication reserves its writer before allocating primary keys for bulk
  Sense/Question inserts; PostgreSQL retains sequence-backed ordered RETURNING.
- Each Question artifact stores its own answer, option IDs, distractors and
  explanations. Consumers must withhold answers on the learner-facing wire;
  `grade_answer` receives saved option IDs or free text, not an answer supplied
  by the browser. Learning history/mastery belongs to the consumer.
- A pending sense relation points at a Word. `resolve_relations` includes every target
  Sense with matching POS, then asks one typed Choice per link; `no_candidate` means no
  match. Both target gloss and source/relation direction must fit. Resolution is explicit
  and can become pending again if target meanings
  change. It does not run during Word generation or reading.
- `DecisionConfig.threshold` is a caller-provided inclusive acceptance boundary
  for both decision-model yes-probabilities and Choice confidence. A
  lower-confidence Choice delegates the original whole request to
  the official System One adapter if configured. Grading consumes the same
  threshold for Noul outcomes; fuzzy lexical search has an unrelated score.
- The caller supplies `generate(..., example_count=5)` per call and coordinates overlapping
  requests, retries, batching and user state. Single operations publish
  atomically. Read models are dataclasses; model outputs are Pydantic.
- The sole migration `20260930_base` bootstraps an empty generated schema only.
  Table definitions are frozen in the revision; existing databases require explicit
  backed-up conversion/parity verification before an operator replaces their stamp.
  No old upgrade history or automatic schema mutation remains. The baseline includes
  scope indexes and statement-level PostgreSQL invalidation;
  neutral and no-op edits retain the existing namespace/invalidation semantics.
  Pending edges are still locked, and apply still locks first then reads evidence
  in a separate READ COMMITTED statement snapshot.
  The `db_schema` Alembic option and constructor parameter
  name an existing PostgreSQL schema for migration and runtime connections. Never point the
  migration config at Cambridge. Postgres tests require an explicitly disposable
  `LEXI_TEST_PG_URL`.

Library configuration is explicit: `LLMConfig` / `DecisionConfig` carry provider
key, base URL and model. Decision fallback reuses LLM credentials/endpoint with a
separate model ID. No library or migration code reads environment variables or
loads `.env`; examples alone parse `LLM_*` and `DECISION_*`. DB/source/schema and
threshold are passed as CLI parameters there. There is no `from_settings()` or
constructor-level content-count configuration.

PostgreSQL handles lexical fuzzy retrieval and scoring through three GIN
`pg_trgm` indexes (lemma, alias, Sense form); B-tree pattern indexes handle
prefixes. Fuzzy `%` filtering uses a transaction-local threshold of 0.3 and
database deduplication before LIMIT. No Python similarity engine or full-Word
scan fallback exists. SQLite is a lexical-only test backend. Pattern matching
uses indexed heads and a bounded matcher separately from trigram suggestions.

Sense Linking finds distinct pending targets by ordered index seeks, then probes a bounded
edge prefix per eligible target and merges by public edge ID. Deferred targets
remain pending and become eligible after publication; no persistent cursor skips
them. The target-driven work scales with distinct pending targets, not all Words
or all deferred edges. It is not a constant-time guarantee for an arbitrarily
large number of distinct pending targets.

## Diagnostic grading

Seven executable JSON-e decision templates live beside Questions and relations;
original annotated designs are retained in `docs/decision/`. Single-word grading
separates task fit from spelling, then searches top-1 Word only when appropriate.
Definition grading resolves intended meaning before accuracy/coverage. Usage gates
on `used` before separate diagnostics against the saved meaning anchor. No inline
grading rubric or legacy `correct|typo|incorrect` result remains. Incomplete evidence
or provider limits produce explicit errors, never silently truncated inventories.
Calibration requires a labeled dataset; none has been supplied or evaluated.

The historical implementation is preserved in `archive/v1/` and is not bundled
in the wheel. See [the codebase map](./codebase-summary.md) for current file owners.
