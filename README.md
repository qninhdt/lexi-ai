# Lexi-AI

A lazy-generation English learner's dictionary library. It synthesizes dictionary
entries with an LLM **on demand**, anchored to Cambridge and WordNet for
hallucination control, and caches results in a local database so repeat lookups
cost zero tokens.

## Features

- **Lazy lookup** — the first lookup of a word spends tokens to synthesize a full
  entry (senses, examples, CEFR levels, aliases, related words); every lookup
  after is a free cache hit. Surface variants (case, diacritics, US/UK spelling,
  `{sb}`/`{sth}` placeholders) fold to one entry via a single normalization key.
- **Reference-anchored generation** — entries are synthesized from Cambridge +
  WordNet anchors, never copied, to keep the LLM honest.
- **Word enrichment** — each entry carries learner-dictionary labels (guideword,
  grammar, register, connotation, collocations) and word-reference links
  (word-family, confused-with), all emitted in the same LLM call.
- **Topic tags & semantic search** — browse words by open-vocabulary topic tags,
  or rank senses by meaning with local embeddings (optional extra).
- **Question engine** — turn a generated word into vocabulary questions across
  four formats and grade answers. Each format is a self-contained plugin owning
  its own generation, grading, and persistence; the engine is a pure dispatcher.
  Covers rule-based and LLM-based generation and grading (see below).
- **Portable storage** — one schema runs on both SQLite and Postgres (portable
  column types only, no JSONB/ARRAY/native enum).

## Install

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management.

```bash
uv sync                      # create .venv and install runtime + dev deps
uv sync --extra embeddings   # optional: local sense embeddings (torch, ~200MB)
```

## Usage

```python
import asyncio
from lexi_ai import Lexicon

async def main():
    lex = Lexicon.from_settings()   # reads LEXI_* env / .env
    await lex.init()

    # Search (free) → generate (spends tokens once) → cached thereafter.
    results = await lex.search("serendipity")
    entry = await lex.generate(results[0])
    print(entry.display, entry.senses[0].definition)

    # Question engine: generate + grade vocabulary questions from a done word.
    questions = await lex.questions.generate(entry, formats=["contextual_mcq"], n=1)
    score = await lex.questions.grade(questions[0], answer=2)
    print(score.correct, score.score)

asyncio.run(main())
```

Configuration is env-driven (prefix `LEXI_`): `LLM_BASE_URL`, `LLM_API_KEY`,
`LLM_MODEL`, `DB_URL`, `CAMBRIDGE_DB_PATH`. Copy `examples/.env.example` to `.env`
to get started. The model is **never hardcoded** — it comes only from
`LEXI_LLM_MODEL`.

### Question formats (v1)

| Format | Answer kind | Generator | Grader | Persists |
|--------|-------------|-----------|--------|----------|
| `definition_mcq` | single choice | rule | rule (index) | no |
| `cloze` | text span | rule | rule (`match_key`) | no |
| `contextual_mcq` | single choice | LLM | rule (index) | yes |
| `use_in_sentence` | free text | rule | LLM (rubric) | no |

Adding a format is one plugin class + one registry line — no engine edit.

## Examples

Runnable end-to-end trials live in [`examples/`](examples/README.md) (they hit a
live LLM on first run). For instance:

```bash
uv run python examples/01_lookup_word.py serendipity
uv run python examples/12_question_engine.py eloquent
```

## Development

```bash
uv run pytest             # full test suite (no live LLM — fake runnables)
uv run ruff check .       # lint
uv run ruff format .      # format
```

The suite is hermetic: no network, no live LLM calls. See
[`docs/system-architecture.md`](docs/system-architecture.md) for the design.

## License

[MIT](LICENSE)
