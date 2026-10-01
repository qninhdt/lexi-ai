# Lexi-AI codebase map

The wheel builds **`lexi_ai`**, exporting **`Lexicon`**. Historical code, tests
and examples are isolated in `archive/v1/` and are not imported by the package.

| Path | Responsibility |
|------|----------------|
| `lexi_ai/__init__.py`, `api.py` | Public `Lexicon` class, lazy provider creation, lifecycle |
| `config.py`, `inference/config.py` | Lexical bounds and explicit LLM/Decision provider settings; counts belong to generation calls |
| `models.py`, `text.py`, `patterns.py`, `vocab.py` | Detached read models, lexical rules and controlled vocabulary |
| `references/cambridge.py`, `references/wordnet.py` | Read-only selected source, optional WordNet evidence |
| `words/search.py`, `words/generate.py`, `words/schemas.py` | Source-neutral search, selected Word generation and structured output |
| `themes/service.py`, `themes/namespace.py` | Theme metadata, exact namespaces and separately persisted themed meaning/examples |
| `questions/generate.py`, `questions/schemas.py`, `questions/grade.py` | Seven question tasks, saved artifacts and grading |
| `relations/resolve.py`, `relations/storage.py`, `relations/invalidation.py` | Complete-inventory Sense Linking, evidence revalidation and invalidation |
| `inference/` | Shared transports, configuration and Jinja/JSON-e renderers; no domain tasks or prompts |
| `translation/generate.py`, `translation/storage.py` | Target-markup preprocessing, translation and exact-text cache |
| `schema.py` | Relational generated-dictionary schema and domain DDL hooks |
| `db/` | Generic sessions, SQL collections, bulk insertion, pagination and random slots; no domain dependencies |
| `migrations/`, `alembic.ini` | One squashed fresh-schema baseline; no legacy upgrades |
| `words/storage.py`, `questions/storage.py` | Fresh relational reads/writes and narrow grading projections; no stored JSON mirror |
| `questions/positions.py`, `db/random.py` | Dense Question banks and generic random-slot selection |
| `db/bulk.py`, `words/storage.py` | Generic multi-row insertion and domain content publication |
| `words/indexes.py` | Three PostgreSQL-native GIN trigram indexes |
| `*/prompts/`, `docs/decision/` | Domain-owned runtime prompts and retained supplied annotated designs |
| `tests/` | Provider-fake, SQLite, wheel and opt-in disposable PostgreSQL tests |

Start with [README.md](../README.md) for the public API and
[system-architecture.md](./system-architecture.md) for data boundaries.
