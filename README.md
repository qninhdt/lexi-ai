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
  grammar, register, connotation, collocations, domain, usage note) and
  word-reference links (word-family, confused-with, hypernym, hyponym), all emitted
  in the same LLM call.
- **Inflection forms** — each sense carries its complete grammatical paradigm
  (`run` → ran/running/runs; `good` → better/best), emitted per POS by the LLM and
  surfaced on `SenseView.forms`. Example sentences tag the target word with its
  inflection (`<t inf="past">glistened</t>`) for display highlighting and cloze
  blanking; `parse_marked_example`/`strip_markup` read the tags.
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
- **Question engine** — prepare, retrieve, and evaluate persisted vocabulary
  questions through five registered types. Plugin identity (`type_id`) is separate
  from the UI contract (`render_format`); level 0 is exposure and levels 1–4 are
  assessments. Preparation is best-effort, retrieval is exact and never generates,
  and evaluation reports `graded` or `pending`.
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
from lexi_ai import LexiconEngine, LexiconReader

async def main():
    # Two facades, one capability boundary. The reader can only read; the engine is
    # the only object that can change a row or spend a model call. A read-only
    # process constructs the reader alone and cannot generate by accident.
    work = LexiconEngine.from_settings()   # reads LEXI_* env / .env
    await work.init()
    read = LexiconReader.from_settings()

    # Search (free) → generate (spends tokens once) → cached thereafter.
    results = await read.search("serendipity")
    entry = await work.generate(results[0])
    print(entry.display, entry.senses[0].definition)

    # Question engine: inspect capabilities, prepare persisted assessments,
    # retrieve one exact question, then evaluate by durable question id.
    from lexi_ai import PrepareDemand

    question_types = work.question_types()
    sense_id = entry.senses[0].sense_id
    report = await work.prepare_questions(
        entry.word_id,
        [PrepareDemand(sense_id=str(sense_id), difficulty_level=1, expected_count=1)],
    )
    question = await read.retrieve_question(
        sense_id,
        difficulty_level=1,
        excluded_ids=frozenset(),
        type_id="definition_mcq",
    )
    if question is not None:
        # Grading a rubric type needs the judge, so it goes through the engine.
        evaluation = await work.evaluate_answer(
            question.question_id, question.payload["correct_index"]
        )
        print(question.type_id, question.render_format, evaluation.status)

    # Themes: author a voice once (LLM-expanded if description/tone are omitted,
    # generated in-line the first time a word is fetched under it), read the overlay.
    theme = await work.create_theme("Pirate", "narrate like a salty pirate")
    themed = await work.generate(results[0], theme=theme.key)
    print(themed.senses[0].definition)                   # restyled definition

    # Cached assets (reference-addressed by sense id): a repeat call is free.
    print(await work.translate_sense(sense_id, "vi"))     # real, LLM-backed
    # TTS is real when LEXI_TTS_* is configured (OpenAI-compatible /audio/speech);
    # unconfigured, the stub raises rather than caching fake audio.
    clip = await work.tts_sense(sense_id)                # Asset (file_path to the clip)

asyncio.run(main())
```

`Lexicon` remains available as the composition root — it wires the object graph and
hands out `reader()` and `engine()` — but it exposes no use case of its own.

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
`tts_many`, `get_status_many`) run concurrently and return a
`list[BatchResult]` — one entry per input, in order; a failed item never aborts
the rest (check `result.ok` / `result.value` / `result.error`). Question work uses
`prepare_questions`; persisted assessments are selected with `retrieve_question`
and evaluated with `evaluate_answer`.

`add_examples(sense_id, n=3, theme=None)` appends up to `n` fresh example
sentences to a single sense (neutral, or a themed overlay when `theme=` is set —
the word must already be themed) and returns the updated `SenseView`; it never
overwrites existing examples and never re-embeds. `stats()` returns read-only
dictionary counts (words by status, senses, examples, tags, themes, themed
words, assets by kind, questions).

### Question types

`question_types()` returns the registered capability descriptors. `type_id`
selects generation/evaluation behavior; `render_format` selects the UI payload
contract, so multiple types can share one renderer. Difficulty is explicit:
level 0 is non-assessable exposure and levels 1–4 are assessments.

| Type ID | Render format | Levels | Mode |
|---------|---------------|--------|------|
| `flashcard` | `flashcard` | 0 | exposure |
| `definition_mcq` | `single_choice` | 1 | assessment |
| `contextual_mcq` | `single_choice` | 1–2 | assessment |
| `cloze` | `text_span` | 2–3 | assessment |
| `use_in_sentence` | `free_text` | 3–4 | assessment |

`prepare_questions(word_id, demands)` best-effort creates persisted assessments
and returns produced counts by `(sense_id, difficulty_level)`.
`retrieve_question(...)` performs exact type/level selection, excludes supplied
question IDs, and never generates or falls back. `retrieve_exposure(sense_id)`
builds the level-0 flashcard. `evaluate_answer(question_id, answer)` returns an
`Evaluation` with status `graded` or `pending`; exposure cards are not assessable.

The existing `matching`, `listening`, `spelling`, `pronunciation_mcq`, and
`collocation_fill` plugin files are intentionally unregistered while they await
migration to this contract in a follow-up.

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
