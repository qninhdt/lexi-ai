# Lexi-AI

An async Python library for building English vocabulary-learning applications.
Generate learner-friendly dictionary content from selected reference entries,
create exercises, grade answers, and reuse saved content across requests.

## Features

- **Dictionary:** definitions, examples, pronunciation, inflections, usage patterns,
  collocations, and semantic relations organized by Word and Sense.
- **Search:** lemma, alias, inflection, and expression matching, with Tantivy
  fuzzy suggestions.
- **Exercises:** seven question types with saved answers and explanations.
- **Grading:** separate diagnostics for word choice, definitions, and usage.
- **Themes:** alternative wording and example contexts for the same lexical meanings.
- **Translation:** text translation with a persistent cache.

## Installation

Requires **Python 3.14+**, an OpenAI-compatible LLM provider, and a generated-content
database. Tantivy provides native search for both PostgreSQL and SQLite. Provider
credentials are needed for generation and inference, not stored reads or data import.

Install from a local checkout:

```bash
python -m pip install .
```

The required project-owned reference dataset is distributed separately as SQLite.
`import_reference()` downloads the pinned GitHub release into the standard user cache,
checks its SHA-256, and imports it into PostgreSQL schema `lexi_reference`. Repeated
imports use the installed dataset without requiring network access or a local file.
`import_reference("/path/reference.sqlite")` supplies an offline snapshot instead.
SQLite consumers read the downloaded reference.sqlite directly; only SQLite accepts
a constructor `reference_path`. PostgreSQL uses `import_reference(path)` for offline setup. The wheel does not
bundle either dataset. WordNet is optional supporting evidence.

The source checkout includes `data/reference.sqlite` and the optional generated
dictionary `data/content.sqlite` through Git LFS; run `git lfs pull` to download
them. The generated snapshot contains dictionary content without Questions and
has a matching `data/content.sqlite.sha256` checksum.

Cambridge and WordNet remain separate internal reference providers. Search returns
unconsumed Cambridge identities as public `REFERENCE` hits; word generation uses
Cambridge plus optional WordNet evidence. Inventory groups evidence from two
anonymized dictionaries into meanings useful for A1–C2 learners preparing for
IELTS/TOEIC. Meanings must have clear learning value for everyday, workplace or
general academic English; infrequent uses with limited practical value are omitted.
Dictionary presence or possible use outside a specialist field alone is insufficient.
It generates the Word identity and lexical-family relations, and fixes each
Sense's definition, POS and supporting references. Sense enrichment runs in
parallel using only Word identity and the fixed definition/POS, without reference
payloads or a repeated target. References and source pronunciation are preserved
locally for publication. The default request is five examples per Sense; validation
keeps usable examples and rejects empty output. Target tags belong only in examples;
patterns, collocations, usage notes and relation glosses are plain text. Patterns use
the supported slots in `lexi_ai/vocab.py`, including `{done}` (past-participle phrase),
`{adj}` (adjective phrase) and `{adv}` (adverb phrase). Slot matching is bounded text
matching, not grammatical analysis. Raw provider data is private.
Reasoning models can reserve up to 32,768 completion tokens through
`LLMConfig.max_completion_tokens` or the CLI's `LLM_MAX_COMPLETION_TOKENS`; the default
remains 4,096. Reasoning consumes part of that output budget.

`search(query, include_reference=True)` returns one ranked `SearchResult.items`
list. Each hit has `kind` (`WORD` or `REFERENCE`), `match_kind` and `matched_surface`.
`MatchKind` is `EXACT`, `PREFIX`, `SUBSTRING` or `FUZZY`, in that relevance order.
Exact lemma, alias, saved inflection and fully anchored licensed pattern matches
are all `EXACT`. Sort by match class, generated word before reference, then matched
surface origin (lemma, alias/alternative, form, pattern) for every match class.
Remaining ties use descending Tantivy score, normalized display and stable ID.
Reference citation/display surfaces are lemmas; alternatives are aliases and
reference inflections are forms. A form such as `began` can select reference
`begin` before that Word has been generated. Patterns
with slots match only fully anchored `EXACT` queries. Fuzzy matching is disabled
below three characters, allows distance one for 3–5, and distance two from six
characters, including adjacent transpositions. One strongest hit is retained per
identity; the default 30-result limit applies to the combined list. References
linked through `WordSource` to a `DONE` word are excluded from the index.

Call `await lexicon.start()` before searching or grading. Startup builds one in-memory
Tantivy index. Publication updates only the committed word and removes its consumed
reference from search, then publishes a new reader; existing readers keep their snapshot.
Search performs no database reads, timed refresh or result caching. Explicit dataset
imports happen before startup. Out-of-process database changes require restarting the
reader; Pycil uses one API process for search and generation.

`generate_word(target)` takes only the first search item. An `EXACT` `WORD` is reused;
an `EXACT` `REFERENCE` is generated using its selected display identity.
An empty list or a suggestion-only result raises `no exact search match`.
Thus `running` reuses a saved `run` with that form, without inference.
Prefix/substring/fuzzy suggestions remain selectable in the UI but do not
implicitly substitute another word. UI callers can pass an explicit `reference_id`.
Requested missing theme content can still require inference on a reused word.

Generated words, senses and questions can be distributed as `content.sqlite`.
`import_content("/path/content.sqlite")` imports them into an initialized PostgreSQL
`lexi` schema; `questions=False` imports only dictionary content. Import reference first.
IDs and foreign keys are preserved, identical rows are reused, conflicting content aborts the transaction,
and ID sequences are advanced. Content import does not invoke inference. The artifact
must use the current native migration head and have no pending WAL sidecar.

`import_content()` without a path downloads `content.sqlite` and verifies its matching
`content.sqlite.sha256` from release `content-data-v1`. That optional release must be
published after building content; until then, supply a local artifact.

### Database setup

For a **new PostgreSQL dictionary database**, apply the packaged migrations once:

```bash
lexi db init --db 'postgresql+asyncpg://user:password@localhost/lexicon'
```

PostgreSQL uses the `lexi` schema by default. The database role needs permission to
create that schema and install `pg_trgm`. For a custom schema, pass `--db-schema`
to the migration CLI and the same `db_schema` to `Lexicon`.
The initial baseline requires an empty generated-content schema.
The same Lexi migration history owns the PostgreSQL reference schema. Constructing
`Lexicon` does not run migrations or download data; call `import_reference()` after Python migration setup; `lexi db init` does both.

## CLI

The installed `lexi` command and `python -m lexi_ai` share the same implementation.
Generate a Word from a string or a UTF-8 TXT word list, or Questions for a saved
Sense ID. Choose a SQLite file or PostgreSQL URL as the output:

```bash
lexi generate word bank --output content.sqlite
lexi generate word --word-list words.txt --output content.sqlite
lexi generate question 1 --type DEFINITION_TO_WORD --count 8 --output content.sqlite
lexi generate word --word-list words.txt --output 'postgresql://user:password@localhost/lexicon'
lexi import content.sqlite --into 'postgresql://user:password@localhost/lexicon'
```

Generation initializes current storage. Word generation loads the required reference
and runs up to 128 words concurrently; `--max-concurrency` changes that limit.
The question command takes a saved Sense ID, one question type and the number of
new Questions to generate; it never generates a Word or missing theme content.
For `DEFINITION_TO_WORD`, `WORD_TO_DEFINITION` and `WORD_TO_USAGE`, CLI defaults
are 1 Question and 5 distractors. Other types default to 8 Questions and 3 distractors.
Explicit `--count` and `--distractor-count` override those defaults. The Python
`generate_questions` API still takes both counts explicitly.
Both commands return full saved artifacts. All provider requests share the Lexicon
instance's limit of 128; separate instances have separate limits. Failed Words are
reported with a nonzero exit while successfully committed content is preserved.

To generate Words plus all seven Question types with resumability and tqdm progress:

```bash
uv run python scripts/generate-content.py --word-list /path/common-words.txt
```

Run from the checkout root. Defaults are `data/reference.sqlite`, `data/content.sqlite`,
32 concurrent words and at most 128 provider requests, 1 Question with 5 distractors per fixed-stem type,
and 8 Questions with 3 distractors per remaining type, for each Sense. `--count`
sets the desired total for those four context types. Each Word starts its Question banks immediately, with all types
running concurrently. Run the same command again to reuse Words and fill only missing
Questions. Use `--word-concurrency` to change the number of simultaneous Words;
`--max-concurrency` limits provider requests, including parallel Sense enrichment.
To generate only the first 100 inputs through both Word stages:

```bash
uv run python scripts/generate-content.py --word-list scripts/1000-most-comon-words.txt --words-only --limit 100 --word-concurrency 32 --max-concurrency 128
```

Completed Words are committed individually and reused on the next run.
See [CLI recipes](examples/README.md) for interruption handling.

SQLite generation closes the database, checkpoints WAL and writes a standalone file
plus `.sha256`. Import copies SQLite content into PostgreSQL without inference;
identical rows are reused and conflicts roll back. `--words-only` omits questions.
Use `lexi import reference.sqlite --kind reference --into POSTGRESQL_URL` for an
offline reference import, or omit the source to download the required reference.

The CLI reads provider settings from `.env` in the current working directory, or
`--env-file PATH` on a command. Process environment overrides that file. The Python
API never reads environment configuration. Read/import commands need no provider
credentials. Run `lexi --help` or any command with `--help`; the complete workflow
examples are in [CLI recipes](examples/README.md).

## Quickstart

Set `LLM_API_KEY` in your application's environment, then run:

```python
import asyncio
import os

from lexi_ai import Lexicon, LLMConfig, QuestionType, ResponseFormat


async def main():
    lexicon = Lexicon(
        db_url="postgresql+asyncpg://user:password@localhost/lexicon",
        db_schema="lexi",
        llm_config=LLMConfig(
            api_key=os.environ["LLM_API_KEY"],
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
        ),
    )
    try:
        await lexicon.import_reference()
        await lexicon.start()
        matches = await lexicon.search("bank", include_reference=True)
        for hit in matches.items:
            print(hit.kind, hit.match_kind, hit.matched_surface)

        word = await lexicon.generate_word("bank", example_count=3)
        for sense in word.senses:
            print(sense.pos, sense.definition.content)

        questions = await lexicon.generate_questions(
            word.senses[0].id,
            QuestionType.DEFINITION_TO_WORD,
            count=2,
            distractor_count=3,
        )
        grade = await lexicon.grade_answer(questions[0].id, ResponseFormat.SINGLE_WORD, "bank")
        print(grade)
        print(await lexicon.translate_text("I went to the bank.", "vi"))
    finally:
        await lexicon.close()


asyncio.run(main())
```

`generate_word()` returns a complete generated/reused `Word`, including its lemma
and senses. `generate_questions(sense_id, question_type, count, distractor_count=3)`
returns the new saved `list[Question]`. Stored reads never generate content.

**Question objects contain answers and explanations.** Consumers choose the visibility
policy: self-learning applications may deliver saved keys for offline practice; exam
applications should conceal them. Shuffling options must preserve their saved IDs.

## Questions and grading

| Question type | Response formats |
| --- | --- |
| `DEFINITION_TO_WORD` | `SINGLE_CHOICE`, `SINGLE_WORD` |
| `CONTEXT_TO_WORD` | `SINGLE_CHOICE`, `SINGLE_WORD` |
| `CLOZE_TO_WORD` | `SINGLE_CHOICE`, `SINGLE_WORD` |
| `WORD_TO_DEFINITION` | `SINGLE_CHOICE`, `SHORT_ANSWER` |
| `WORD_TO_USAGE` | `SINGLE_CHOICE`, `SHORT_ANSWER` |
| `DIALOGUE_COMPLETION` | `SINGLE_CHOICE` |
| `MEANING_IN_CONTEXT` | `SINGLE_CHOICE` |

Import `QuestionType`, `ResponseFormat`, `TargetPlacement` and the lexical/diagnostic
enums from `lexi_ai`; inspect native capabilities through `QUESTION_FORMATS` or
`ALLOWED_PAIRS`, rather than maintaining a separate consumer vocabulary. Canonical
public/storage tokens are uppercase. The `20261009_base` baseline initializes fresh generated dictionaries and PostgreSQL
reference storage. Historical revisions are unsupported.

`grade_answer()` returns task-specific diagnostics:

- **Word choice:** `task_fit`, `spelling_error`, `sense_id`.
- **Definition:** `sense_id`, `accuracy`, `coverage`.
- **Usage:** `used`, `meaning`, `form`, `construction`, `collocation`, `appropriacy`.

Saved option IDs and normalized exact saved word answers are graded locally.
Other answers use model inference. Diagnostics may be `None` when an earlier
stage cannot identify a meaning or detect target use. Model judgments are not
guaranteed correctness or learner-mastery scores.

## Configuration

`Lexicon(max_concurrency=128)` limits concurrent provider operations across its
word, question, theme, translation, grading and relation calls. Schedule independent
calls with asyncio; SQLite serializes writes while provider requests run concurrently.

Pass configuration explicitly; the library does not load `.env` files.
`LLMConfig` supports provider URL/model, timeout, output-token limit, temperature,
reasoning effort, and bounded retries. Set `structured_outputs=False` for prompted
JSON output with local repair and schema validation.

Grading and Sense Linking use the LLM by default. To use a native decision model,
pass a `DecisionConfig` with credentials, URL, model, and confidence threshold.
Both operations accept these modes:

| Mode | Behavior |
| --- | --- |
| `llm_fallback` | Use the decision provider if configured; low-confidence Choices use the LLM. Without a decision provider, use the LLM directly. |
| `decision_only` | Use only the configured decision provider. |
| `llm_only` | Use only the LLM. |

Add `with_usage=True` to AI operations to receive `(result, list[TokenUsage])`.
Usage is aggregated by reported model ID; cache hits return an empty list and
unreported token counts remain `None`.

## Integration

`Lexicon` exposes Word/Sense reads, Theme management, question generation/retrieval,
grading, relation resolution, and translation-cache management. Lists support
cursor pagination. See the [CLI recipes](examples/README.md) for each workflow.

Themes use separate content namespaces: `theme=None` selects neutral content;
requesting a Theme never silently substitutes neutral content. Translation caches
use the exact unwrapped input text and target language.

The application owns scheduling and coordination of overlapping operations.
`resolve_relations()` explicitly processes a global page of eligible links; Word
generation does not resolve them automatically. If you supply an SQLAlchemy
`AsyncSession` instead of `db_url`, you own its transaction and lifecycle.

## Development

```bash
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run lint-imports
uv build
```

PostgreSQL tests require `LEXI_TEST_PG_URL` pointing to a **disposable** database.

## License

[MIT](LICENSE) covers this project's code and project-owned reference dataset.
Third-party dependencies retain their own licenses.
