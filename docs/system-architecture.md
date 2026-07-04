# Lexi-AI System Architecture

Lazy-generation English learner's dictionary. Entries are synthesized by an LLM
**on demand**, anchored to Cambridge + WordNet for hallucination control, and
cached in a local database so repeat lookups cost zero tokens.

## The one invariant that matters

`normalize.match_key(s)` is a lossy lookup key computed by **one function**, used
on both the write path (indexing generated words/aliases) and the read path
(resolving user input). If the two paths ever diverged, lookups would miss
forever. Only `persistence/repository.py` writes keys; `api.py` reads them; both
import the same `match_key`. Every surface variant of a word — case, diacritics,
whitespace, `{sb}`/`{sth}` placeholders, US/UK spelling — folds to the same key.

`render(norm)` is the inverse-ish display function (`{sb}` → "somebody"). Display
is always rendered at read time; there is **no** `display` column.

## Package layout

```
lexi_ai/
  normalize.py      match_key / render (pure, zero-I/O; THE invariant)
  constants.py      controlled vocabularies (single source: ORM + LLM schema)
  config.py         pydantic-settings (LLM creds, two DB locations)
  db.py             async engine + session_scope; SQLite FK pragma
  models.py         SQLAlchemy 2.0 async ORM (portable types only)
  read_models.py    dataclass views returned to callers
  references/       read-only anchors
    cambridge.py    Cambridge SQLite (mode=ro), fetch + candidates + phrase_titles
    wordnet.py      nltk WordNet via asyncio.to_thread
    loader.py       ReferenceBundle = Cambridge + WordNet
  generation/       LLM synthesis
    schemas.py      Pydantic GeneratedResult (strict enums from constants)
    prompts.py      system prompt + tier rubric + split-vs-alias rule + formatter
    generator.py    LangChain 1.x ChatOpenAI.with_structured_output; retry
  theming/          restyle a done entry's senses in a named voice
    schemas.py      Pydantic ThemedResult (definition + fresh in-voice examples)
    prompts.py      voice system prompt + neutral-facts formatter (NO neutral examples)
    generator.py    ThemedGenerator — ChatOpenAI.with_structured_output; retry
  assets/           content-addressed derived-asset cache
    repository.py   AssetRepository + content_hash / normalize_asset_params
    translate.py    Translator (real, LLM-backed) — cache-first
    tts.py          TTSProvider protocol + StubTTSProvider (raises; real drop-in later)
  persistence/
    repository.py   THE write path: upsert, dedup, stub-link, cefr, themes, error path
  questions/        generate + grade vocabulary questions from a done entry
    base.py         plugin contract: QuestionContext, QuestionFormat, registry
    distractors.py  best-effort wrong-option ladder (semantic -> topic)
    schemas.py      GeneratedMCQ / Judgment (llm) + per-format payload validators
    scoring.py      shared async grade helpers (single_choice / text_span / rubric)
    formats.py      the four format plugins + registry wiring
    repository.py   QuestionRepository (the questions write path; JSON at boundary)
    engine.py       QuestionEngine — the dispatcher over plugins
  api.py            Lexicon.get() — the public lazy-lookup surface
  prep/
    phrase_overlap.py  Phase-7 one-off: classify Cambridge phrase_titles
```

## Lazy lookup flow (`Lexicon.get`)

```
input → match_key() → resolve against words.match_key AND word_aliases.alias_match_key
  ├─ N words   → Candidates       (homograph; no disambiguation)
  ├─ 1 done    → Entry            (cache HIT — zero LLM calls)
  ├─ 1 pending → generate → Entry (stub promoted on demand)
  └─ 0         → ReferenceBundle → generate → persist → Entry
```

Misses run the pipeline **once per key**, guarded by a per-key `asyncio.Lock`
plus a DB double-check inside the lock. This is a **library, single-process**
(decision #18); `words.match_key` UNIQUE + an `IntegrityError`/SAVEPOINT re-fetch
is the durable backstop.

## Data model

Fourteen tables in the *generated* DB (separate from read-only Cambridge, no
cross-DB FK — decision #14):

- `words` — `norm`, `match_key` (UNIQUE), `entry_type`, `status`
  (pending/done/error/not_found), `pos`, `cambridge_word_id`.
- `word_aliases` — same-entry surface variants; `alias_match_key` indexed;
  UNIQUE(word_id, alias_match_key).
- `entry_links` — cross-entry relations; `to_word_id` is **always a real FK id**
  (stub-row pattern #11); UNIQUE(from, to, rel_type). `rel_type` includes the two
  word-reference relations `word_family` / `confused_with` (normalized like
  synonyms — see below).
- `senses` — the core (sense-centric #6); `tier`, `cefr_level`, `sense_order`;
  learner-dictionary labels `guideword` (short homograph disambiguator), `grammar`
  (0-3 closed-vocab labels, comma-joined in one column), `register`, `connotation`
  (both closed-vocab enums); plus a best-effort semantic-search vector
  (`embedding` BLOB + `embedding_model` + `embedding_dim`, null until an embedder
  runs — no pgvector, portable).
- `sense_reference` — N-N provenance to a Cambridge sense / WordNet synset
  (may be empty).
- `examples` — per sense.
- `collocations` — per sense; high-frequency partner phrases (make a decision,
  heavy rain), ordered free text, structurally identical to `examples`.
- `tags` — open-vocabulary topic tags; `name` slug + `title` display + `tag_key`
  (UNIQUE, the topic analogue of `match_key`, computed only by the repository).
- `word_tags` — word↔tag join; UNIQUE(word_id, tag_id).
- `questions` — generated vocabulary questions about a word (optionally a sense);
  `format`, `answer_kind`, and a `payload` JSON string (app-serialized, portable
  `Text` — the per-format content, so a new format needs no new table). No UNIQUE
  key: questions are content, not identity. Only questions a plugin chose to
  persist get a row (see below).
- `themes` — user-authored style voices; `name` + `style_prompt` + `theme_key`
  (UNIQUE, the theme analogue of `tag_key` but WITHOUT singularization — a name is
  a proper voice). Addressed BY key at the API, so the key is exposed in the read
  model (unlike `tag_key`).
- `themed_senses` — a neutral sense restyled in a theme's voice; `definition` +
  UNIQUE(sense_id, theme_id). Cascades from either the neutral sense or the theme.
- `themed_examples` — fresh in-voice examples per themed sense (mirrors `examples`).
- `assets` — content-addressed derived-asset cache; identity is
  UNIQUE(content_hash, kind, params). `text_value` for inline results
  (translation), `file_path` for binary clips (TTS, relative to the cache dir). No
  FK to senses/words — the source is identified by CONTENT, not location.

**Themes** (user-authored voices, overlay model): a theme restyles an entry's
definitions + examples in a named voice ("Harry Potter", "humorous"). The neutral
`words`/`senses`/`match_key` invariant is untouched — themed content lives in
`themed_senses`/`themed_examples` and is generated by `generate_theme` AFTER the
neutral content, in one LLM call anchored to the neutral sense FACTS (definition,
pos, guideword, tier) but NOT the neutral examples (those are re-authored
in-voice). `get(word_id, theme_key=…)` overlays themed def+examples per-sense,
falling back to neutral where a theme hasn't run; `theme_key=None` is the neutral
entry unchanged. Only definition + examples are themed this round; all other
fields stay neutral, and themed text is never embedded or semantically searched.

**Cached assets** (content-addressed): derived text (translation, TTS) is keyed by
`content_hash(source_text)` + a normalized `params` token, computed by ONE function
on read and write (the `match_key`/`tag_key` discipline). This is why "each theme
has its own translation/TTS" falls out for free: themed vs neutral text hash
differently → distinct assets, identical text dedups. `translate(text, lang)` is
real (LLM-backed, cache-first — a repeat call spends zero tokens); `tts(...)` is an
interface + stub this round (the stub raises rather than caching fake audio, so a
stubbed miss leaves no row/file). Translation results live inline in `text_value`;
TTS clips write to `LEXI_ASSET_CACHE_DIR` sharded by hash prefix, with the row
storing a RELATIVE path (file written before row; a row implies a file).

**Word enrichment** (learner-dictionary content, LLM-authored): each generated
word gets seven enrichments emitted in the same LLM call as senses — synthesized
from the Cambridge/WordNet anchors, not copied. Two kinds: **word-references**
(`word_family`, `confused_with`) NAME a lemma, so they are normalized through the
existing `related[]` → `entry_links` path (match_key stub-rows + dedup) and appear
in `Entry.links` by `rel_type`; **sense labels** (`guideword`, `grammar`,
`register`, `connotation`, `collocations`) LABEL a sense and live on `senses` /
the `collocations` child table. Closed-vocab enums live in `constants.py` (single
source for ORM + schema); free text is control-char sanitized on the write path.
All best-effort — a sense with no enrichment still persists `done`.

**Topic tags** (open-vocabulary, LLM-authored): each generated word gets 1-3 tags
emitted in the same LLM call as senses. Consistency without embeddings: the full
existing tag vocab is injected into every generation prompt for reuse, a
deterministic `tag_key` (lowercase / singular / diacritic- and control-folded)
dedups case/plural variants on the write path, and resolve-or-create under the
UNIQUE key keeps one row per tag (title set once, first-seen). Browse via
`list_tags()` (live member count) and `words_by_tag(tag)` (exact filter, resolved
through `tag_key`). Both FREE — 0 LLM calls.

## Question engine

`Lexicon.questions` turns a `done` entry into vocabulary questions and grades
answers. It *manages* questions (create / read / delete / grade); it does not
*use* them — rotation, quiz sessions, SRS, and progress are the application's job.

**Three axes wired through `answer_kind`.** A **format** declares an `answer_kind`
(what an answer looks like: `single_choice` / `text_span` / `free_text`); a
**generator** turns an entry into a question; a **scorer** turns `(question,
answer)` into a score. A format is not bound to a backend — a rule and an LLM are
just two ways to implement the same interface.

**One plugin per format; the engine is a dispatcher.** Each format is a single
plugin that owns BOTH halves — `async generate` and `async grade` — so the two
cannot drift. The engine looks a plugin up by `format`, hands it a
`QuestionContext` of capabilities (the entry, a distractor provider, optional
bound `llm`/`judge` runnables, and a narrow `store` façade), and `await`s it. The
engine never inspects which backend a plugin uses, and grading dispatches by
`format` to the plugin, which delegates to a shared helper keyed to its
`answer_kind` — so the same `grade_single_choice` grades any single-choice
question regardless of who generated it.

**A plugin owns its persistence.** There is no persistence rule in the engine: a
plugin that wants a row calls `ctx.store.insert(...)` itself, inside its own
`generate`; every other plugin returns ephemeral questions (`id=None`). The engine
cannot tell the difference. `store` is the CRUD façade of `QuestionRepository`
(the questions write path), so plugins get a DB door, not a session. Grading needs
no DB — a freshly generated, never-stored question grades fine.

The seed set proves the abstraction by covering every backend combination:

| Format | answer_kind | Generator | Grader | Persists |
|--------|-------------|-----------|--------|----------|
| `definition_mcq` | `single_choice` | rule | rule (index) | no |
| `cloze` | `text_span` | rule | rule (`match_key`) | no |
| `contextual_mcq` | `single_choice` | llm | rule (index) | yes |
| `use_in_sentence` | `free_text` | rule | llm (rubric) | no |

`contextual_mcq` (llm-generated, rule-graded) and `use_in_sentence` (rule-generated,
llm-graded) are the cross-axis proofs. Adding a format is one plugin class + one
`register(...)` line; the registry validates the format↔answer_kind coupling at
import time (a mis-wire is an import error). Payload is app-level JSON in a `Text`
column (the one deviation from native typing), (de)serialized only at the
repository boundary, which rejects an embedded NUL so it round-trips safely on
Postgres. Distractors are best-effort (semantic neighbours, then shared topic
tags); an MCQ degrades to fewer options rather than fabricating. The LLM plugin and
judge are injectable, so the whole subsystem tests with fake runnables and zero
network.

**Portability:** only `Text`/`String`/`Integer`/`DateTime`/`LargeBinary` — no
JSONB/ARRAY/native
ENUM. Verified by compiling every table's DDL against both the SQLite and
Postgres dialects (`tests/test_models.py::test_schema_compiles_on_both_dialects`).

## Key design decisions

- **Split vs alias** (#16/#17): a Cambridge page bundling independent lemmas
  splits into N `words` rows (correctness — else a member lookup misses and
  duplicates); same-meaning surface variants (log in/log on, color/colour) stay
  ONE word + aliases. Enforced by the prompt; schema allows `units: list`.
- **cefr Cambridge-first** (#13): when a sense references a Cambridge sense that
  carries a CEFR value, that wins over the LLM's guess. The repository takes a
  `cambridge_cefr` map (built by `api.py` from the bundle) — no source coupling.
- **Stub rows** (#11): a mentioned related word becomes a `pending` `words` row
  immediately, so links are real ids and the stub doubles as the lazy-gen queue.
- **Async safety:** the repository never touches relationship collections on a
  *persistent* object (that triggers a lazy-load outside greenlet context);
  children are cleared with Core `delete()` and re-inserted with explicit FK ids.
  Read models are built inside the session via `selectinload`.

## Configuration

Env vars (prefix `LEXI_`): `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`,
`LLM_TEMPERATURE`, `DB_URL` (generated DB), `CAMBRIDGE_DB_PATH` (read-only source).
Assets/themes: `ASSET_CACHE_DIR` (TTS clip dir, default `./lexi-assets`),
`TRANSLATE_MODEL` (optional translation model override, falls back to `LLM_MODEL`),
and `TTS_BASE_URL`/`TTS_API_KEY`/`TTS_MODEL`/`TTS_VOICE`/`TTS_FORMAT` (reserved for
the TTS provider — defined but unused this round, since TTS ships as stub only).

## Testing

256 tests, `uv run pytest`. No live LLM calls (fake runnables). Reference and
phrase-overlap tests run against the real Cambridge `./data` and skip if absent.
Themes and cached assets add `tests/test_themes.py` (theme_key folding, dedup,
schema compile), `tests/test_theming.py` (themed generation + read overlay with a
fake generator, count-mismatch guard, per-sense fallback), and
`tests/test_assets.py` (content-hash addressing, param normalization,
resolve-or-create, translation cache hits, TTS stub-raises + real-provider
round-trip).
The suite includes regression tests for a multi-agent review pass: Postgres
NUL-safety of `match_key`, concurrent-insert savepoint recovery, the
Cambridge-first CEFR contract (`sense#42` vs bare `42`), per-key lock eviction,
and the word-enrichment write→read spine (word-reference normalization/dedup,
closed-vocab enum rejection, collocation ordering + sanitization, eager-load).
The question engine is covered end-to-end: each format's generate + grade, the
two cross-axis proofs, plugin-owned persistence (the engine stays blind to it),
the registry coupling guard, a one-line new-format extensibility proof, and
payload round-trip (unicode + NUL rejection).
