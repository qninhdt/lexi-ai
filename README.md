# Lexi-AI

An on-demand English learner's dictionary **Python library**. Search the read-only
Cambridge SQLite snapshot for available entries, generate one selected Word into a
separate database, and reuse stored content on later reads. The public import is
`from lexi_ai import Lexicon`. The previous implementation is archived under
`archive/v1/`; there are no compatibility imports or aliases.

## Install

The active package requires **Python 3.14+**. Development and CI pin **CPython
3.14.7** in `.python-version`; Python alpha/beta/release-candidate builds are not
selected. Runtime/dev dependencies are resolved in `uv.lock`, and the build
backend uses `uv_build` 0.12.21 or a compatible 0.12 patch.

```bash
uv python install 3.14.7
uv sync --locked            # PostgreSQL/SQLite drivers, provider SDKs and dev tools
```

Use uv **0.12.21+** for the current Python download catalog. If an older local uv
cannot download the pinned interpreter, run
`uv tool run --from uv==0.12.21 uv python install 3.14.7` without replacing the
system-wide uv binary. This upgrade does not change the system Python or archive.

The Cambridge SQLite snapshot is `data/cambridge.db`, tracked through Git LFS and
opened read-only. After cloning, install Git LFS and run `git lfs pull` if the file
was not downloaded automatically. Do not use an LFS pointer as a SQLite database.
Never run migrations against this snapshot. Generated dictionary DBs remain ignored.
Production generated dictionaries use PostgreSQL with `pg_trgm`. Apply Alembic migrations using
`lexi_ai/alembic.ini` from the installed package and a generated-database URL;
Pass `db_schema="my_schema"` to `Lexicon` for an existing non-default PostgreSQL
schema, and set Alembic's `db_schema` option to the same name for migrations. For tests,
`from lexi_ai.schema import Base; await lexicon.db.create_schema(Base.metadata)` creates
a fresh generated database directly. The migration role needs permission to install
`pg_trgm` if it is absent. The sole active revision is `20260930_base`, a fresh-schema
baseline with frozen table definitions and domain index/trigger installation.
It refuses to bootstrap over existing tables. Previous migration histories are no
longer supported; **do not run `upgrade head` on an existing older dictionary**.
Back up first, compare its schema/indexes/triggers with this baseline, and perform any
required conversion explicitly. Only after verified parity may an operator replace
the old version stamp (`stamp --purge head`); stamping does not convert tables, repair
data or install triggers. No database reset, migration or stamping happens automatically.
SQLite generated databases are lexical-only development/test stores:
exact/prefix/pattern search is supported, fuzzy suggestions are not.
The baseline runs online; offline SQL export is not supported.

Example one-time migration (pass the URL of the **generated** DB, not
Cambridge; create a named PostgreSQL schema first if using one):

```python
from pathlib import Path

from alembic import command
from alembic.config import Config
import lexi_ai

cfg = Config(str(Path(lexi_ai.__file__).parent / "alembic.ini"))
db_url = "postgresql+asyncpg://user:password@localhost/lexicon"
cfg.set_main_option("sqlalchemy.url", db_url.replace("%", "%%"))
# cfg.set_main_option("db_schema", "my_schema")  # optional existing schema
command.upgrade(cfg, "head")
```

## Use

See [examples/README.md](examples/README.md) for eight standalone public API examples.

```python
from lexi_ai import DecisionConfig, Lexicon, LLMConfig

lexicon = Lexicon(
    "postgresql+asyncpg://user:password@localhost/lexicon",
    "data/cambridge.db",
    llm_config=LLMConfig(
        api_key="your-llm-key", base_url="https://api.openai.com/v1", model="gpt-4o",
    ),
    decision_config=DecisionConfig(
        threshold=0.8, api_key="your-decision-key",
        base_url="https://api.typesafe.ai", model="jev-latest",
    ),
    decision_fallback_model="gpt-4o",
)
try:
    matches = await lexicon.search("bank", include_available=True)
    for hit in matches.available:
        print(hit.available_id, hit.display)
    # Only selected, requestable handles trigger generation; reads never generate.
    available_id = input("Selected Cambridge available_id: ")
    word = await lexicon.generate(available_id, example_count=3)
    neutral = await lexicon.get_word(word.id)

    theme = await lexicon.create_theme("pirate", "Pirate", "A nautical speaking voice")
    themed = await lexicon.generate(available_id, theme=theme.key, example_count=5)
    questions = await lexicon.generate_questions(
        themed.senses[0].id, "definition_to_word", 2,
        distractor_count=3, theme=theme.key,
    )
    # Question objects include the answer. Keep them on a trusted server;
    # present only prompt text and opaque choice IDs to the learner.
    submitted_option_id = input("Option ID returned by your learner-facing UI: ")
    grade = await lexicon.grade_answer(questions[0].id, "single_choice", submitted_option_id)
    links = await lexicon.resolve_relations()  # explicit; never run by get_word()
    translated = await lexicon.translate_text("bank", "vi")  # same text/language is cached
finally:
    await lexicon.close()
```

The caller owns retries, concurrency, and learner progress; the library does not
schedule or batch calls across requests. `theme=None` is neutral. Themed content
and questions are separate exact namespaces, not overlays or fallback reads.
Each Sense has one `definition` per namespace; only example counts are configurable
(the `generate(..., example_count=5)` default is five per Sense). Definition and Example
remain relational rows, not stored Word JSON. A call that creates both neutral and
themed content uses its count for both; generate neutral first if different counts
are needed for the themed namespace. Counts are positive integers, not constructor settings.
Stored themed content is reused on
subsequent requests even if the caller changes the desired counts.

`DecisionConfig.threshold` is **one configurable inclusive boundary** for decision-model
yes-probabilities and Choice confidence. Lower-confidence Choices
fall back through the official System One adapter when a fallback model is set;
`grade_answer` uses the same threshold for decision-model booleans. Pass credentials,
base URL and model through `DecisionConfig` / `LLMConfig`. Decision fallback uses
the LLM key/base URL with the separately selected `decision_fallback_model`.

Grading uses domain-owned JSON-e templates and returns separate diagnostics:

See [the implemented Decision design](docs/decision/README.md) for stage gates,
output contracts, relation direction and fallback behavior.

- Choice/single-word: `task_fit`, `spelling_error`, `sense_id`. Task fit and spelling
  are independent. Only a fit answer without spelling error searches the top-ranked
  Word and selects from **all** its Senses; missing mapping does not change correctness.
  Saved option IDs and exact saved single-word answers avoid provider calls.
- Definition: `sense_id`, `accuracy`, `coverage`. Resolve the intended owner-Word
  Sense first, then judge accuracy/coverage; no identified Sense yields null diagnostics.
- Usage: `used`, `meaning`, `form`, `construction`, `collocation`, `appropriacy`.
  If unused, diagnostics are null; otherwise judge them separately against the saved
  meaning anchor.

Relation resolution considers **all** same-POS target Senses and requires both target
gloss compatibility and directional source-relation validity. Incomplete evidence or
provider limits return per-edge errors, not truncated inventories or negative decisions.

The library and migration runner never read `.env` or environment variables.
`Lexicon.from_settings()` and `ContentCounts` have been removed; there are no aliases.
Missing credentials raise an error only when a provider is actually needed, rather
than falling back to SDK environment configuration. Provider defaults are explicit:
LLM `gpt-4o` at `https://api.openai.com/v1`, Decision `jev-latest` at
`https://api.typesafe.ai`. Injected providers remain supported and caller-owned.
Only examples load `.env`, using the `LLM_*` and `DECISION_*` groups; DB/source/schema,
threshold and per-generation counts are CLI parameters, not `.env` settings.

Question types: `definition_to_word`, `context_to_word`, `cloze_to_word`,
`word_to_definition`, `word_to_usage`, `dialogue_completion`,
`meaning_in_context`. `dialogue_completion` accepts `target_placement="dialogue"`
or `target_placement="options"`. For `definition_to_word`, `word_to_definition`,
and `context_to_word`, the correct option is attached from trusted Word/Sense data;
the model supplies only task content where needed, an explanation, and distractors.
Every artifact is saved once and supports its documented
response formats; `retrieve_question` reads only stored artifacts. There is no
TTS, automatic source generation, hidden learner/mastery state, or compatibility
with the archived implementation or former package API.

Full Word/Sense reads aggregate independent relational children in one statement.
Generation loads only the selected Sense; grading projects only candidate meanings.
Question-only dense slots support uniform random retrieval with an indexed maximum
and indexed slot lookup, without `COUNT(*)` or loading the bank. SQL triggers maintain
both namespace-wide and type-scoped slots on inserts, deletes, and scope changes.
Explicit scope ordering lets neutral banks reuse the slot indexes for maximum lookup.
Sense Linking pages reuse candidates per distinct target/POS; readers transfer neutral target
evidence once per distinct target. Pattern pages remain bounded to 200 patterns and
load forms once per distinct Sense in that page. These are query-time projections,
not persistent caches or materialized documents.

For bounded list reads use `list_questions(..., limit=100, after_id=last_id)` or
`list_translations(limit=100, after_id=last_id)`. Theme lists retain key ordering:
`list_themes(limit=100, after_key=last_key)`. Limits are 1–500; omitting `limit`
preserves the existing list-all behavior. Cursors are exclusive, with no OFFSET.

`translate_text` validates and removes target-expression tags before sending text
to the model. Its cache uses the exact unwrapped text and target language: tagged
and plain versions share a cache entry; whitespace remains significant.

## Search

Normalized lemma, alias and Sense-form keys use B-tree exact/prefix indexes and
three PostgreSQL GIN trigram indexes. Results prioritize exact lemma, alias, form,
licensed pattern, prefix, then fuzzy suggestions, with one best hit per Word and
at most 30 results. PostgreSQL filters fuzzy matches with `%` at threshold `0.3`,
scores with `similarity()`, and deduplicates before limiting results. There is no
Python fuzzy matching, arbitrary raw-candidate cap, GiST KNN or SQLite fuzzy
fallback. Queries of 1–2 normalized characters use exact/prefix only.

Trigram similarity is not edit distance or semantic equivalence and can miss
short-word typos. Pattern matching is a separate bounded, Sense-scoped operation.
GIN cost depends on candidate selectivity; `LIMIT 30` does not promise that only
30 index matches are processed. Available Cambridge results remain separate.

`inference/` contains shared model infrastructure, not domain use cases.
Question/Word generation and schemas stay in `questions/` and `words/`;
Theme and translation logic stay in `themes/` and `translation/`.

Packaged domain-owned Jinja prompts under `lexi_ai/*/prompts/` declare
`{# system #}` followed by `{# user #}`. The
renderer splits those sections before interpolation, keeping reference/user
content out of the system role.

## Verify

```bash
uv run pytest -q tests
uv run ruff check lexi_ai tests
uv run lint-imports
uv build
```

`LEXI_TEST_PG_URL` opts into disposable PostgreSQL search, migration and concurrency
checks. They are skipped without an explicit disposable URL; `LEXI_REQUIRE_PG=1`
makes its absence an error in CI. See [the technical design](docs/lexi-ai-technical-design.md)
for the current domain contract. The old code/tests/examples are preserved under `archive/v1/` and are
excluded from packaging and the active test suite.

For the reproducible 100,000-Word retrieval probe, set the same disposable URL and
run `uv run python tests/benchmark_postgres_search.py --source /path/to/cambridge.db
--output /tmp/opencode/lexi-search-benchmark.json`. The probe creates and removes
an isolated schema and reports latency, index usage, buffers, and actual query plans;
it is evidence for this environment, not a production latency SLO.
See [search verification](docs/search-verification.md) for measured results,
planner observations, and the limits of the corpus probe.
