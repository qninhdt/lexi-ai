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
- **Selective anchoring** — senses (definitions + examples) are synthesized from
  Cambridge + WordNet anchors, never copied, to keep the LLM honest; IPA
  pronunciation is hard-anchored from Cambridge (per POS); semantic relations are
  LLM-generated, not anchored.
- **Pronunciation** — each sense carries per-POS IPA (`ipa_uk` / `ipa_us`),
  anchored from Cambridge and surfaced on `SenseView`.
- **Word enrichment** — each entry carries learner-dictionary labels (guideword,
  grammar, register, connotation, collocations) and word-reference links
  (word-family, confused-with, hypernym, hyponym), all emitted in the same LLM call.
- **Topic tags & semantic search** — browse words by open-vocabulary topic tags,
  or rank senses by meaning with local embeddings (optional extra).
- **Themes** — restyle an entry's definitions and examples in a named voice
  ("Harry Potter", "humorous") authored via `create_theme`. Themed content
  overlays the neutral entry (the canonical `match_key` invariant is untouched)
  and is generated once after the neutral content, then cached — the app picks
  one active theme like a light/dark-mode switch.
- **Cached assets** — reference-addressed cache for derived content: **translation**
  (real, LLM-backed) and **text-to-speech** (real, OpenAI-compatible). Identity is
  the source reference `(source_kind, source_id, kind, params)`, plus a stored
  `content_hash` verified on read — so a regenerated or reused source yields a clean
  miss (never stale content), and a repeat call spends zero tokens.
- **Question engine** — turn a generated word into vocabulary questions across
  seven formats and grade answers. Each format is a self-contained plugin owning
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

    # Themes: author a voice once (LLM-expanded if description/tone are omitted,
    # generated in-line the first time a word is fetched under it), read the overlay.
    theme = await lex.create_theme("Pirate", "narrate like a salty pirate")
    themed = await lex.generate(results[0], theme=theme.key)
    print(themed.senses[0].definition)                   # restyled definition

    # Cached assets (reference-addressed by sense id): a repeat call is free.
    sense_id = entry.senses[0].sense_id
    print(await lex.translate_sense(sense_id, "vi"))     # real, LLM-backed
    # TTS is real when LEXI_TTS_* is configured (OpenAI-compatible /audio/speech);
    # unconfigured, the stub raises rather than caching fake audio.
    clip = await lex.tts_sense(sense_id)                 # Asset (file_path to the clip)

asyncio.run(main())
```

Configuration is env-driven (prefix `LEXI_`): `LLM_BASE_URL`, `LLM_API_KEY`,
`LLM_MODEL`, `DB_URL`, `CAMBRIDGE_DB_PATH`. Copy `examples/.env.example` to `.env`
to get started. The model is **never hardcoded** — it comes only from
`LEXI_LLM_MODEL`.

Asset and theme knobs (all `LEXI_`-prefixed):

- `ASSET_CACHE_DIR` — where TTS clips are written (default `./lexi-assets`);
  translation results live in the DB.
- `TRANSLATE_MODEL` — optional per-task model override for translation; falls
  back to `LLM_MODEL` when empty.
- `TTS_BASE_URL`, `TTS_API_KEY`, `TTS_MODEL`, `TTS_VOICE`, `TTS_FORMAT` — the
  OpenAI-compatible TTS provider. When a key is set, `TTS_BASE_URL` must be
  `https://` (or a loopback host) so the key is never sent in cleartext. Leave
  them unset and TTS falls back to a stub that raises rather than caching fake
  audio.

### Managing & batch

Every resource has get/list/delete alongside create — `get_theme`/`update_theme`/
`delete_theme`, `delete_entry`/`list_entries`/`list_entries_by_tag`,
`rename_tag`/`delete_tag`/`merge_tags`, `get_asset`/`list_assets`/`delete_asset`/
`purge_assets`. Bulk variants (`generate_many`, `get_many`, `translate_many`,
`get_status_many`, `lex.questions.grade_many`) run concurrently and return a
`list[BatchResult]` — one entry per input, in order; a failed item never aborts
the rest (check `result.ok` / `result.value` / `result.error`).

### Question formats

| Format | Answer kind | Generator | Grader | Persists |
|--------|-------------|-----------|--------|----------|
| `definition_mcq` | single choice | rule | rule (index) | no |
| `cloze` | text span | rule | rule (`match_key`) | no |
| `contextual_mcq` | single choice | LLM | rule (index) | yes |
| `use_in_sentence` | free text | rule | LLM (rubric) | no |
| `matching` | matching | rule | rule (permutation) | no |
| `listening` | single choice | rule (TTS) | rule (index) | yes |
| `spelling` | text span | rule (TTS) | rule (`match_key`) | no |

`listening` and `spelling` synthesize an audio clip via the configured TTS
provider; with no TTS configured they degrade to no questions rather than
failing. Adding a format is one plugin class + one registry line — no engine edit.

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
