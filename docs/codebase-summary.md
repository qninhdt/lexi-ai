# Codebase Summary

Navigation map for `lexi_ai`. For *why* the layers are shaped this way, read
[system-architecture.md](./system-architecture.md); this file is the *where*.

- **Package:** `lexi_ai`, 12.4K LOC across 96 modules
- **Tests:** `tests/`, 9.9K LOC across 34 modules, 565 tests
- **Python:** >= 3.10 · **Build:** `uv_build` · **Lint:** ruff · **Boundaries:** import-linter

## Layers

Dependencies point inward. `contracts` and `domain` know nothing about the ORM;
`application` reaches persistence only through ports. Three import-linter
contracts in `pyproject.toml` enforce exactly that.

```
consumer
   │
   ▼
facades/          the public API — reader.py (209) + engine.py (293)
   │
api.py (276)      composition root: wires everything, owns no use case
   │
   ▼
application/      use cases; talk to ports, never to the ORM
   │
   ▼
domain/           models, ports, errors, hashing — no infrastructure, no SQLAlchemy
   ▲
   │
infrastructure/   the adapters that satisfy those ports
contracts/        dependency-free wire types shared with plugins
```

## Entry points

| Path | LOC | Role |
|------|-----|------|
| `api.py` | 276 | `Lexicon` — the ONLY entry point. `Lexicon.from_settings()`, then `.reader()` / `.engine()`. Wiring only; no use case of its own (locked by a test). |
| `facades/reader.py` | 209 | `LexiconReader` — free reads. Cannot mutate, cannot call a provider. |
| `facades/engine.py` | 293 | `LexiconEngine` — everything that can generate, write or spend money. |
| `contracts/` | 223 | Wire types for third-party question plugins. Dependency-free by contract. |

`LexiconReader`/`LexiconEngine` have no `from_settings` of their own on purpose:
two graphs would mean two DB engines and two `SingleFlight` registries, so one
word could generate twice.

## `application/` — use cases

| Module | LOC | Owns |
|--------|-----|------|
| `generation.py` | 254 | The lazy-generation use case (the core flow). |
| `generation_writer.py` | 88 | The write transaction for a generated entry; records failures then re-raises. |
| `enrichment.py` | 250 | Post-commit additive work: examples, embeddings, relation resolution, backfills. |
| `themes.py` | 198 | Themed overlays (restyled definitions/examples). |
| `dictionary.py` | 147 | Entry reads, deletes, purge. |
| `search.py` | 140 | Lexical `search` + `semantic_search`. |
| `assets.py` | 141 | Translation text and TTS clips. |
| `questions.py` | 100 | Question preparation and retrieval. |
| `tags.py` | 42 | Topic tags. |
| `question_ports.py` | 69 | Ports the question engine needs. |
| `single_flight.py` | 40 | One in-flight generation per key, process-wide. |
| `batching.py` | 37 | Batch-result plumbing. |

## `domain/` — the middle

`models.py` (176) domain models · `ports.py` (306) every port the application
depends on, incl. `UnitOfWork` and `VectorIndex` · `questions.py` (74) question domain
· `errors.py` (5) · `hashing.py` (15) content hashing.

## `infrastructure/` — adapters

```
db/
  models.py        (573)  SQLAlchemy models — the single schema source
  repositories/           entry, sense (527), word (330), theme (289), tag (181), stats
  uow.py                  UnitOfWork implementation
  mappers.py       (197)  ORM rows -> domain/read models
  sanitize.py             write-path input scrubbing
  types.py                custom column types
  asset_gc.py             orphaned-asset collection
vectors/
  lancedb_index.py (179)  durable ANN index (LEXI_VECTOR_BACKEND=lancedb)
  memory_index.py   (68)  exact scan, non-durable — the hermetic test default
  validation.py     (18)  shared vector/metadata checks
providers.py     (176)  LLM / translator / TTS provider factories
question_engine_factory.py  caches the question engine per capability context
```

## Supporting packages

| Package | Role |
|---------|------|
| `questions/` | Question engine (`engine.py`), base plugin (`base.py`), 5 built-in types in `types/`, scoring, dedup, distractors, render, repository. |
| `generation/` | LLM output schemas (`schemas.py`, 404), generator, word-sense disambiguation. |
| `references/` | Read-only reference data: `cambridge.py` (425), `wordnet.py`, `loader.py`. |
| `assets/` | `repository.py` (408) content-addressed cache, `translate.py`, `tts.py`. |
| `theming/` | Themed-overlay generation and its schemas. |
| `prep/` | `phrase_overlap.py` — pre-generation phrase checks. |
| `migrations/` | Alembic. Baseline `20260724_01`, head `20260724_02`. |
| `prompts/` | Jinja prompt templates. |

## Top-level modules

`constants.py` (515) shared enums/tables · `normalize.py` (261) lemma
normalization and `render()` · `read_models.py` (259) caller-facing shapes ·
`llm.py` (224) the OpenAI-compatible seam · `embeddings.py` (147) local encoder,
lazily imported · `config.py` (87) one flat `Settings`, env prefix `LEXI_` ·
`db.py` (137) engine/session factories · `markup.py` (70) · `vectors.py` (31)
dependency-free cosine.

## Where to change what

| Task | Start here |
|------|-----------|
| Add a caller-facing method | `facades/reader.py` or `facades/engine.py`, then the service in `application/` |
| Change what the LLM returns | `generation/schemas.py` + the prompt in `prompts/` |
| Change the schema | `infrastructure/db/models.py`, then a new Alembic revision |
| Add a vector backend | one module in `infrastructure/vectors/` + a branch in `build_vector_index` |
| Add a question type | `questions/types/`, register in the `lexi_ai.question_types` entry-point group |
| Add a provider | `infrastructure/providers.py` |
