# Lexi-AI — live API trials

Runnable trials that exercise `lexi_ai` end-to-end against a real
OpenAI-compatible LLM. Each script hits the live API on the **first** creation of a
word and reads from the local cache afterward.

## Setup

```bash
cp examples/.env.example .env          # from the repo root; .env is gitignored
uv sync                                # install deps (once)
```

The trials read all config from `.env` via the `LEXI_` env prefix — **the model
is never hardcoded** (`LEXI_LLM_MODEL`). Defaults in `.env.example`:

| Var | Meaning |
|-----|---------|
| `LEXI_LLM_BASE_URL` | proxy endpoint |
| `LEXI_LLM_API_KEY` | key |
| `LEXI_LLM_MODEL` | model id (`cd/gpt-5.5`) |
| `LEXI_LLM_TEMPERATURE` | sampling temperature |
| `LEXI_LLM_REASONING_EFFORT` | `minimal`/`low`/`medium`/`high` (read by `_common.py`) |
| `LEXI_DB_URL` | generated-dictionary cache (SQLite) |
| `LEXI_CAMBRIDGE_DB_PATH` | read-only Cambridge source (`./data`) |

Run every script **from the repo root** so `.env` and `./data` resolve.

## Public API (suggestion-centric)

A raw string is ambiguous (one string → many Cambridge words, or a typo), so you
never create an entry from it directly. The flow is **resolve → pick → create**:

```python
sugs  = await lex.resolve("serendipity")   # FREE — ranked Cambridge matches
entry = await lex.get(sugs.items[0])        # create-or-return the chosen suggestion
```

| Method | Cost | Purpose |
|--------|------|---------|
| `resolve(raw) -> Suggestions` | FREE | ranked Cambridge matches (exact 1.0 → fuzzy), each flagged `done`/`pending`/`absent` |
| `get(sug) -> Entry` | may generate | create-or-return the entry for a chosen suggestion |
| `get_many(sugs) -> list[Entry]` | may generate | concurrent, order-preserving, dedups |
| `peek(sug) -> Entry \| None` | FREE | cached entry or `None` |
| `exists(sug) -> bool` | FREE | is it generated? |
| `status(sug) -> str \| None` | FREE | `done`/`pending`/`error`/`None` |
| `add(raw) -> Entry` | may generate | custom entry for a word Cambridge lacks; never overwrites |

The demos 01–07 use a `lookup(lex, raw)` helper in `_common.py` that wraps
"resolve then get the top match" to stay short; **08 shows the real two-step flow.**

## Trials

| Script | Shows |
|--------|-------|
| `01_lookup_word.py [word]` | full lookup; 1st = generate, 2nd = free cache hit |
| `02_alias_resolution.py [canonical] [variant]` | US/UK variants collapse to one entry (`color`/`colour`) |
| `03_placeholder_lookup.py "[phrase]"` | idioms normalize to `{sb}`/`{sth}` tokens; variant phrasings share a key |
| `04_concurrent_lookups.py [word] [n]` | N simultaneous misses generate exactly once (per-key lock) |
| `05_related_graph.py [word]` | related words persist as `pending` stubs, generated on demand |
| `06_interactive_repl.py` | type words by hand; watch them cache |
| `07_inspect_matching.py [word]` | how a string maps to Cambridge + WordNet anchors |
| `08_resolve_and_pick.py [query]` | the real `resolve → pick → get(sug)` flow + `peek`/`exists`/`status`/`add` |
| `09_semantic_search.py` | rank generated senses by **meaning** via local embeddings (`semantic_search`, `backfill_embeddings`) |
| `10_topic_tags.py` | open-vocabulary **topic tags** per word; browse via `list_tags` / `list_entries_by_tag` |
| `11_word_enrichment.py [word]` | learner-dictionary **enrichments**: guideword, grammar, register, connotation, collocations + word-family / confused-with links |
| `12_question_engine.py [word]` | **question engine**: generate + grade questions across 4 formats; llm questions persist for 0-token reuse |

```bash
uv run python examples/08_resolve_and_pick.py serendipity
uv run python examples/01_lookup_word.py serendipity
uv run python examples/02_alias_resolution.py color colour
uv run python examples/03_placeholder_lookup.py "look after somebody"
uv run python examples/04_concurrent_lookups.py ephemeral 5
uv run python examples/05_related_graph.py happy
uv run python examples/07_inspect_matching.py happy
uv run python examples/06_interactive_repl.py
```

## Semantic search (example 09)

Ranking generated senses by **meaning** needs sense embeddings, computed locally
by a `transformers` model (the chat proxy has no embeddings endpoint). Install the
optional extra once, then run:

```bash
uv sync --extra embeddings                    # torch + transformers (~200MB, CPU)
uv run python examples/09_semantic_search.py  # weights (~90MB) download on first run
```

Embeddings are **best-effort**: without the extra, generation still works and
`semantic_search` returns `[]`. After installing, `backfill_embeddings()`
vectorizes any senses generated earlier. Tune via `LEXI_EMBEDDING_*` in `.env`.

## Topic tags (example 10)

Every generated word gets 1-3 broad **topic tags** the LLM invents (no predefined
list). Consistency is enforced without embeddings: the full existing tag vocab is
injected into each prompt for reuse, a deterministic `tag_key` dedups case/plural
variants (`Cars`/`car`/`CAR` → one tag), and resolve-or-create keeps one row per
tag. Each tag has a short `name` and a human `title` (set once, first-seen wins).

```bash
uv run python examples/10_topic_tags.py
```

Browse is FREE (no LLM): `list_tags()` enumerates every topic with its live member
count; `list_entries_by_tag("business")` returns the generated words carrying it
(resolved via `tag_key`, so `"Business"` and `"business"` hit the same tag).

## Word enrichment (example 11)

Every generated word carries seven learner-dictionary **enrichments**, emitted in
the same LLM call as its senses (no extra requests) and anchored to
Cambridge/WordNet — the model **synthesizes** them, it does not copy the source.
They split by one test — does the field NAME a lemma or LABEL this sense?

- **word-references** (`word_family`, `confused_with`) NAME a lemma, so they are
  **normalized** like synonyms: they ride `related[]` → `entry_links` and surface
  in `entry.links` by `rel_type`, deduped to one real `words` row per lemma
  (`"Happiness"` and `"happiness"` fold to one).
- **sense labels** LABEL this sense: `guideword` (homograph disambiguator),
  `grammar` (0-3 closed-vocab labels), `register`, `connotation` — columns on
  `senses`; `collocations` — a child table mirroring `examples`.

```bash
uv run python examples/11_word_enrichment.py bank
```

All fields are best-effort: a sense the model leaves unmarked still persists
`done`, with empty/`None` enrichments.

## Question engine (example 12)

Turns a done word into vocabulary questions across four formats and grades
answers. The three axes — format, generator, scorer — are wired through
`answer_kind`, and the engine is a pure dispatcher: each format is a
self-contained plugin that owns its own generation, grading, and persistence.

- `definition_mcq` (rule) — "which word means <definition>?"
- `cloze` (rule) — fill the blank in a real example sentence
- `contextual_mcq` (llm) — MCQ from a novel context; **persists** for reuse
- `use_in_sentence` (rule prompt, **llm-graded** by a rubric)

Only `contextual_mcq` talks to an LLM and persists — it calls the store itself, so
a re-run lists it back at zero token cost. The other three are ephemeral. Grading
dispatches to the plugin: MCQs grade deterministically by index/value; the
free-text answer is judged against a rubric by the LLM.

```bash
uv run python examples/12_question_engine.py eloquent
```

Distractors are best-effort (semantic neighbours, then shared topic tags), so an
MCQ degrades to fewer options rather than fabricating a wrong answer.

## Notes

- **Cost:** only the first lookup of a given word spends tokens. Delete the cache
  to start fresh: `rm -f examples-lexi.db*`.
- **`resolve()` is FREE but not instant:** it fuzzy-scans the 113k Cambridge
  headwords (~1s cold) — no LLM, no tokens. Exact matches still score 1.0 on top.
- **Why `_common.py` rebuilds the generator:** the library's default
  structured-output method (`json_schema`) is not strictly enforced by this
  proxy, so `_common.py` uses `method="function_calling"` plus a
  `reasoning_effort`. Model, key, URL, and temperature still come only from env.
- **First run downloads nothing** — WordNet + Cambridge data are already local
  (`~/nltk_data`, `./data`).
