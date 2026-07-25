# Code Standards

What this codebase actually does — verified against the source, not aspirational.
Layer rules live in [system-architecture.md](./system-architecture.md); the file
map is in [codebase-summary.md](./codebase-summary.md).

## Tooling

| Concern | Tool | Command |
|---------|------|---------|
| Lint | ruff (`E,F,I,UP,B`), line length 100, target `py310` | `uv run ruff check lexi_ai tests examples` |
| Format | ruff format | `uv run ruff format lexi_ai tests examples` |
| Boundaries | import-linter, 3 contracts | `uv run lint-imports` |
| Tests | pytest, `asyncio_mode = "auto"` | `uv run pytest -q` |
| Migrations | alembic 1.18.5 (pinned) | `uv run alembic -c lexi_ai/migrations/alembic.ini upgrade head` |

All four run in CI (`.github/workflows/test.yml`) on push and PR to `main`.
`asyncio_mode = "auto"` means an `async def test_` needs no decorator.

## Layering (enforced, not advisory)

The three import-linter contracts in `pyproject.toml`:

1. `lexi_ai.contracts` imports nothing from domain, application, infrastructure,
   the ORM or SQLAlchemy — it is the wire surface plugins share.
2. `lexi_ai.domain` imports no infrastructure and no SQLAlchemy.
3. `lexi_ai.application` and `lexi_ai.api` never import
   `infrastructure.db.models` directly. Reaching the ORM *through* the unit of
   work is the point; a direct import is a second persistence path.

Breaking one fails CI. These rules are not self-detecting at runtime — a domain
module importing the ORM works perfectly and only shows up later as coupling.

## Types

- Built-in generics and PEP 604 unions: `list[int]`, `str | None`. No `Optional`,
  no `List` (there are zero occurrences of either).
- Ports are `Protocol`s in `domain/ports.py`; adapters live in `infrastructure/`
  and are never imported by name from the application layer.
- `# noqa: ANN001` with a trailing reason where a duck-typed collaborator is
  deliberately untyped (e.g. the reference loader).

## Naming

- Modules `snake_case`; classes `PascalCase`; functions/vars `snake_case`.
- Private helpers and module-level constants that are not API: leading `_`.
- Repositories are named per aggregate: `word_repo.py`, `sense_repo.py`. One
  aggregate per repository — no cross-aggregate queries.
- Services are `<Thing>Service` in `application/`; the caller-facing wrappers are
  the two facades.
- Test modules `test_<area>.py`; test names are full sentences describing the
  behaviour, not the method: `test_semantic_search_raises_when_the_index_is_unreachable`.

## Comments and docstrings

The dominant convention, and the one to follow:

- Every module opens with a one-line summary, then a paragraph on *why* it exists
  and what invariant it protects.
- Docstrings explain the contract and the failure mode, including what a method
  RAISES and what an empty result means.
- Inline comments explain a decision, not the mechanics. Prefer "two reference ids
  fold onto one generated word" over "loop over candidates".
- Never reference plan artifacts, phase numbers or finding codes in code. Plan
  headers get renumbered and the reference rots; the reason must stand alone.
- `# noqa: BLE001` always carries a justification for the broad catch.

## Errors

- Raise on failure. Return an empty result only when the operation ran and found
  nothing — an empty list must never be able to mean "the subsystem is down".
- The one sanctioned swallow is the post-commit generation hook
  (`Lexicon._embed_words`): the entry is committed and the LLM call is paid for,
  so a vector failure must not fail the generation. Tolerance belongs at the call
  site that needs it, never inside the shared service.
- `ValueError` for caller mistakes (unknown id, missing overlay, absent LLM).
- The write transaction records a failure and then re-raises; error recording must
  never mask the cause.

## File size

The project rule is 200 LOC. 19 of 96 modules exceed it, the largest being
`infrastructure/db/models.py` (573) and `repositories/sense_repo.py` (527). Those
are cohesive by aggregate and splitting them would scatter one schema across
files, so they stay. Treat 200 as the trigger for *asking* whether a module is
doing two jobs, not as a hard cap — and do not add a new module over 200 LOC
without that answer.

## Tests

Two tiers, both in `tests/`:

- **Hermetic (default)** — SQLite plus the in-memory vector index, forced by
  `tests/conftest.py`. No network, no LLM, no service container. Must pass on a
  base `uv sync`.
- **Postgres (opt-in)** — `uv sync --extra postgres` and `LEXI_TEST_PG_URL` set to
  a disposable database. Covers what SQLite hides: NUL bytes, `VARCHAR(n)` limits,
  naive datetimes. Gate with `pytestmark = requires_postgres`; use
  `requires_asyncpg_driver` when only the dialect is needed and no server.
  `LEXI_REQUIRE_PG=1` turns a skip into a hard error so CI cannot go green on a
  tier that never ran.

Conventions:

- No mocked database. Tests write through a real engine.
- Fake at the *port*, not the method: `tests/test_services_with_fakes.py` drives
  services with fake ports and no DB at all (25 tests, ~0.05s).
- Providers (LLM/TTS) are always faked; never hit a real endpoint.
- One assertion subject per test. Prefer a new test to a second scenario.
- The adapter contract for vector backends is one parametrized module
  (`test_vector_index.py`) run over every backend, with the exact-scan in-memory
  index as ground truth.

## Migrations

- `infrastructure/db/models.py` is the single schema source; migrations are
  generated from it, and `alembic check` in CI fails if a model change has no
  migration.
- Never edit a pushed revision. Add a new one — anyone who already upgraded would
  otherwise keep the old shape forever.
- Every revision needs a working `downgrade`; the deploy gate runs upgrades
  unattended, so a one-way migration is a one-way door in production.
- Revision ids are `YYYYMMDD_NN`; the filename slug describes the domain change,
  never a plan phase.
- Consumers embed the dictionary in their own schema, so migrations must render
  offline against a non-default schema (`-x schema=…`).

## Configuration

One flat `Settings` (`config.py`, 87 lines) with `env_prefix = "LEXI_"`.
Consumers bind ~16 `LEXI_*` names, so renaming a field is a breaking change for
them. Services receive primitives or narrow ports — do not pass `Settings` into a
service. Nested settings were considered and rejected; the reasoning is recorded
in the redesign plan.

## Commits

Conventional commits, `!` for breaking changes. The body states what changed and
*why the previous behaviour was wrong*, not what the diff shows. No AI
attribution. Formatting-only churn goes in its own `style:` commit so review can
skip it.
