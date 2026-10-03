# Lexi-AI

An async Python library for building English vocabulary-learning applications.
Generate learner-friendly dictionary content from selected reference entries,
create exercises, grade answers, and reuse saved content across requests.

## Features

- **Dictionary:** definitions, examples, pronunciation, inflections, usage patterns,
  collocations, and semantic relations organized by Word and Sense.
- **Search:** lemma, alias, inflection, and expression matching, with PostgreSQL
  fuzzy suggestions.
- **Exercises:** seven question types with saved answers and explanations.
- **Grading:** separate diagnostics for word choice, definitions, and usage.
- **Themes:** alternative wording and example contexts for the same lexical meanings.
- **Translation:** text translation with a persistent cache.

## Installation

Requires **Python 3.14+**, an OpenAI-compatible LLM provider, and a generated-content
database. PostgreSQL with `pg_trgm` supports the full search functionality; SQLite
is suitable for local development without fuzzy search.

Install from a local checkout:

```bash
python -m pip install .
git lfs pull
```

The project-owned reference dataset is `data/cambridge.db`, distributed through
Git LFS and opened read-only. It uses a Cambridge-compatible SQLite schema and is
separate from the database storing generated content. The wheel does not bundle it.

### Database setup

For a **new PostgreSQL dictionary database**, apply the packaged migrations once:

```bash
python -m lexi_ai.migrations upgrade \
  'postgresql+asyncpg://user:password@localhost/lexicon'
```

PostgreSQL uses the `lexi` schema by default. The database role needs permission to
create that schema and install `pg_trgm`. For a custom schema, pass `--db-schema`
to the migration CLI and the same `db_schema` to `Lexicon`.
The initial baseline requires an empty generated-content schema.
Never migrate the reference dataset. Constructing `Lexicon` does not run migrations.

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
        cambridge_path="data/cambridge.db",
        llm_config=LLMConfig(
            api_key=os.environ["LLM_API_KEY"],
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
        ),
    )
    try:
        matches = await lexicon.search("bank", include_available=True)
        for hit in matches.available:
            print(hit.available_id, hit.display)

        # Select an entry from the available results, not an arbitrary identifier.
        if not matches.available:
            print("Stored matches:", matches.words)
            return
        available_id = input("Selected available_id: ").strip()
        word = await lexicon.generate(available_id, target="bank", example_count=3)
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

Generation reuses an already-generated selected entry. Stored reads never generate
content, while each `generate_questions()` call appends new questions.

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
public/storage tokens are uppercase. The single `20261003_base` baseline initializes
fresh generated dictionaries; there is no upgrade path for older schemas.

`grade_answer()` returns task-specific diagnostics:

- **Word choice:** `task_fit`, `spelling_error`, `sense_id`.
- **Definition:** `sense_id`, `accuracy`, `coverage`.
- **Usage:** `used`, `meaning`, `form`, `construction`, `collocation`, `appropriacy`.

Saved option IDs and normalized exact saved word answers are graded locally.
Other answers use model inference. Diagnostics may be `None` when an earlier
stage cannot identify a meaning or detect target use. Model judgments are not
guaranteed correctness or learner-mastery scores.

## Configuration

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
cursor pagination. See the [runnable examples](examples/README.md) for each workflow.

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
