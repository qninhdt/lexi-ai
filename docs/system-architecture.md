# Lexi-AI System Architecture

Lazy-generation English learner's dictionary. Entries are synthesized by an LLM
**on demand**; sense content is anchored to Cambridge + WordNet and IPA is
hard-anchored from Cambridge for hallucination control (semantic relations are
LLM-generated, not anchored), and results are cached in a local database so
repeat lookups cost zero tokens.

## The one invariant that matters

`normalize.match_key(s)` is a lossy lookup key computed by **one function**, used
on both the write path (indexing generated words/aliases) and the read path
(resolving user input). If the two paths ever diverged, lookups would miss
forever. Only the persistence adapter writes keys; `api.py` reads them; both
import the same `match_key`. Every surface variant of a word — case, diacritics,
whitespace, `{sb}`/`{sth}` placeholders, US/UK spelling — folds to the same key.

`render(norm)` is the inverse-ish display function (`{sb}` → "somebody"). Display
is always rendered at read time; there is **no** `display` column.

## Package layout

```
lexi_ai/
  normalize.py      match_key / render (pure, zero-I/O; THE invariant)
  constants.py      controlled vocabularies (single source: ORM + LLM schema)
   config.py         pydantic-settings (LLM credentials and database location)
  db.py             async engine + session_scope; SQLite FK + transaction pragmas
  read_models.py    dataclass views returned to callers
  markup.py         parse/strip the <t inf> example target tags (one reader)
  domain/           technology-free core
    ports.py        repository, unit-of-work, and vector-index Protocols
    models.py       records crossing the persistence boundary (never ORM rows)
    errors.py       failures callers branch on (StaleGenerationError)
    hashing.py      sense_content_hash — content identity of a sense
    questions.py    presented question -> grading spec mapping
  application/      use cases; they own transaction boundaries
    dictionary.py   free reads: entry, senses, status, listings, stats
    search.py       lexical search over references + semantic ranking
    generation.py   generate / generate_many / generate_fenced, single-flight
    generation_writer.py  claim / publish / record-failure, one transaction each
    enrichment.py   added examples, embedding backfill, sense-relation resolve
    themes.py       theme CRUD, restyle-a-word, themed overlays and examples
    tags.py         topic-tag rename / delete / merge
    assets.py       cache-first translation and speech
    questions.py    prepare / retrieve / evaluate over ONE question engine
    question_ports.py  the two narrow seams the question engine consumes
    single_flight.py   per-key async lock registry (collapse duplicate work)
    batching.py     order-aligned concurrent batch execution
  infrastructure/
    providers.py    lazy LLM / WSD / translator / TTS construction from settings
    question_engine_factory.py  builds and caches the reader / worker engines
    vectors/        similarity search, one module per backend
      lancedb_index.py  the default: embedded, on disk, ANN query, no server
      memory_index.py   exact-scan, non-durable — the hermetic test default
      validation.py     the uniform-dimension check every backend owes callers
    db/                 the SQLAlchemy adapter
      models.py       SQLAlchemy 2.0 async ORM (portable types only)
      types.py        Vocabulary / VocabularyList columns enforcing the vocabularies
      sanitize.py     LLM-text cleaning + column caps shared by every repository
      mappers.py      the single ORM <-> domain <-> read-model translation home
      asset_gc.py     collect cached assets for rows about to be deleted
      uow.py          SqlAlchemyUnitOfWork — one session, one commit boundary
      repositories/   one module per aggregate: word, sense, theme, tag, stats, entry
  references/       read-only anchors
    cambridge.py    Cambridge SQLite (mode=ro), fetch + candidates + phrase_titles
    wordnet.py      nltk WordNet via asyncio.to_thread
    loader.py       ReferenceBundle = Cambridge + WordNet
  llm.py            StructuredLLM — wraps openai.AsyncOpenAI chat.completions.parse
  generation/       LLM synthesis
    schemas.py      Pydantic GeneratedResult (strict enums from constants)
    prompts.py      system prompt + tier rubric + split-vs-alias rule + formatter
    generator.py    openai structured-output synthesis; retry
  theming/          restyle a done entry's senses in a named voice
    schemas.py      Pydantic ThemedResult (definition + fresh in-voice examples)
    prompts.py      voice system prompt + neutral-facts formatter (NO neutral examples)
    generator.py    ThemedGenerator — openai structured-output; retry
  assets/           reference-addressed derived-asset cache
    repository.py   AssetRepository + content_hash / normalize_asset_params
    translate.py    Translator (real, LLM-backed) — cache-first
    tts.py          TTSProvider protocol + StubTTSProvider + OpenAICompatibleTTSProvider
  questions/        prepare + retrieve + evaluate persisted vocabulary questions
    base.py         type descriptors, demands, plugin contracts, registry
    distractors.py  best-effort wrong-option ladder (semantic -> topic)
    schemas.py      render-payload validators + structured LLM outputs
    scoring.py      shared evaluation helpers (single choice / text span / rubric)
    formats/        one module per type; five MVP types self-register on import
      _shared.py    cross-type helpers (MCQ build, target blanking, exposure)
      *.py          registered types plus unregistered follow-up candidates
    repository.py   QuestionRepository (durable idempotent question store)
    engine.py       QuestionEngine — prepare/retrieve/evaluate dispatcher
  api.py            Lexicon — the composition root; wires the object graph only
  facades/          THE public API: two capability facades
    reader.py       LexiconReader — free reads (cannot mutate, cannot call a provider)
    engine.py       LexiconEngine — generation, enrichment, curation, assets
  prep/
    phrase_overlap.py  Phase-7 one-off: classify Cambridge phrase_titles
```

## The public API is a capability boundary

`Lexicon` is a composition root, not a service. It owns what must be
process-unique — the database engine, the provider registry, the single-flight
lock registries, the cached question engines — and hands out application services
wired over them. It exposes no use case of its own beyond `init` / `close`
(pinned by `tests/test_facades.py`), so it cannot regrow into a god object.

Callers hold one of two facades, chosen by what they are allowed to do:

| | `LexiconReader` | `LexiconEngine` |
|---|---|---|
| mutates rows | never | yes |
| calls a provider | never | yes |
| needs LLM/TTS credentials | no | yes |
| question grading | provider-free (rubric types degrade) | authoritative |

A read-serving process constructs only the reader and therefore cannot spend a
model call by accident. That is the point of the split: it is enforced by the
absence of the method, not by a runtime flag.

Services are rebuilt per accessor call. They are stateless wiring over a fresh
unit of work, so no caller can accidentally share a session or observe another
caller's transaction; the state that must be shared lives on the `Lexicon`.

## Lazy lookup flow (`LexiconReader.get_entry` / `LexiconEngine.generate`)

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

Fifteen tables in the *generated* DB (separate from read-only Cambridge, no
cross-DB FK — decision #14):

- `words` — `norm`, `match_key` (UNIQUE), `entry_type`, `status`
  (pending/done/error/not_found), `pos`, `cambridge_word_id`.
- `word_aliases` — same-entry surface variants; `alias_match_key` indexed;
  UNIQUE(word_id, alias_match_key).
- `word_relation` — cross-entry relations; `to_word_id` is **always a real FK id**
  (stub-row pattern #11); UNIQUE(from, to, rel_type). `rel_type` includes the
  word-reference relations `word_family` / `confused_with` and the taxonomic
  relations `hypernym` / `hyponym` (all normalized like synonyms — see below).
- `senses` — the core (sense-centric #6); `tier`, `cefr_level`, `sense_order`;
  per-POS IPA pronunciation (`ipa_uk` / `ipa_us`, hard-anchored from Cambridge,
  NUL-sanitized on the write path); learner-dictionary labels `guideword` (short
  homograph disambiguator), `grammar` (0-3 closed-vocab labels, comma-joined in
  one column), `register`, `connotation` (both closed-vocab enums), `domain`
  (subject-area label) and `usage_note` (one-line usage/confusable hint — both
  free text, NUL-sanitized). NO embedding column: sense vectors live in the
  vector index (see below), keyed by sense id.
- `sense_reference` — N-N provenance to a Cambridge sense / WordNet synset
  (may be empty).
- `examples` — per sense.
- `collocations` — per sense; high-frequency partner phrases (make a decision,
  heavy rain), ordered free text, structurally identical to `examples`.
- `sense_forms` — per sense; the inflection paradigm (`inf` label ∈
  INFLECTION_LABELS + `surface`), LLM-emitted per POS (verb → base/past/
  past_participle/present_3sg/ing; noun → plural; adjective → comparative/
  superlative), ordered free text like `examples`. NOT scraped from examples —
  a full paradigm, so grading folds an inflected answer (`ran` for `run`).
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
- `assets` — reference-addressed derived-asset cache; identity is
  UNIQUE(source_kind, source_id, kind, params) — the source row this asset derives
  from, not its content. A stored `content_hash` (NOT part of the identity) is
  verified on read, so a reused/regenerated `source_id` yields a cache miss
  (regenerate), never poisoned content. `text_value` for inline results
  (translation), `file_path` for binary clips (TTS, relative to the cache dir). No
  cross-table FK on `source_id` (the source is a sense, example, or collocation
  depending on `source_kind`) — the read-time hash verify, not a FK, guards
  correctness.

**Themes** (user-authored voices, overlay model): a theme restyles an entry's
definitions + examples in a named voice ("Harry Potter", "humorous"). The neutral
`words`/`senses`/`match_key` invariant is untouched — themed content lives in
`themed_senses`/`themed_examples` and is generated (via `generate(..., theme=)`)
AFTER the neutral content, in one LLM call anchored to the neutral sense FACTS
(definition, pos, guideword, tier) but NOT the neutral examples (those are
re-authored in-voice). `get_entry(word_id, theme=…)` overlays themed def+examples
per-sense, falling back to neutral where a theme hasn't run; `theme=None` is the
neutral entry unchanged. Only definition + examples are themed this round; all
other fields stay neutral, and themed text is never embedded or semantically
searched. Theme management (`get_theme`/`update_theme`/`delete_theme`) is plain
CRUD alongside `create_theme`/`list_themes` — a theme's key is immutable once
created (like `tag_key`); only display fields (name/style_prompt/description/
tone) can be updated.

**Cached assets** (reference-addressed, hash-verified): derived content (translation,
TTS) is keyed by its source reference `(source_kind, source_id)` + `kind` + a
normalized `params` token — the source ROW, not its text. A `content_hash` of the
source text is stored alongside and re-verified on read: if the source was
regenerated (or a `source_id` reused) the hash no longer matches, so the read is a
clean miss (regenerate) rather than serving stale content. This is why "each theme
has its own translation/TTS" still falls out for free — a themed source row is a
distinct `(source_kind, source_id)`. `translate_sense(sense_id, lang)` /
`translate_field(source_kind, source_id, lang)` are real (LLM-backed, cache-first —
a repeat call spends zero tokens); `tts_sense(sense_id)` / `tts_field(...)` POST to
an OpenAI-compatible `/audio/speech` endpoint when `LEXI_TTS_*` is configured, else
fall back to a stub that raises rather than caching fake audio (a stubbed miss leaves
no row/file). Translation results live inline in `text_value`; TTS clips write to
`LEXI_ASSET_CACHE_DIR` sharded by hash prefix, with the row storing a RELATIVE path
(file written before row; a row implies a file). Management is id-based:
`list_assets`/`get_asset`/`delete_asset`/`purge_assets` inspect and prune the cache
(`delete`/`purge` also unlink the backing file). `tts_many(refs, ...)` is the batch
mirror of `translate_many` — order-aligned `BatchResult`s over `(source_kind,
source_id)` refs, cache-first per item, one failure never aborts the rest.

**Word enrichment** (learner-dictionary content, LLM-authored): each generated
word gets a set of enrichments emitted in the same LLM call as senses —
synthesized by the LLM (semantic relations are NOT anchored — Cambridge/WordNet
feed sense content only). Two kinds: **word-references** (`word_family`,
`confused_with`, `hypernym`, `hyponym`) NAME a lemma, so they are normalized
through the existing `related[]` → `word_relation` path (match_key stub-rows +
dedup) and appear in `Entry.links` as a flat list by `rel_type`; **sense labels**
(`guideword`, `grammar`, `register`, `connotation`, `domain`, `usage_note`,
`collocations`, `forms`) LABEL a sense and live on `senses` / the `collocations`
and `sense_forms` child tables. `domain` (subject-area label) and `usage_note`
(one-line usage/confusable hint) are bounded free text like `guideword`; `forms`
is the full inflection paradigm (see below). Closed-vocab enums live in
`constants.py` (single source for ORM + schema); free text is control-char
sanitized on the write path. All best-effort — a sense with no enrichment still
persists `done`.

**Inflection forms & example markup**: the LLM emits each sense's COMPLETE
inflection paradigm per POS directly into `forms` (verb → base/past/
past_participle/present_3sg/ing; noun → plural; adjective → comparative/
superlative) — it is NOT scraped from examples, since one example uses only one
form but the paradigm must be whole. Independently, every example sentence wraps
its target in `<t inf="label">surface</t>` markup: this is a deliberate contract,
not noise — examples are stored WITH the tags intact, and `lexi_ai.markup`
(`parse_marked_example` / `strip_markup`) is the ONE reader (like `match_key` is
the one key function). The cloze plugin blanks the tagged span directly; display
consumers call `strip_markup`. The `forms` surfaces also feed `accepted_forms` in
cloze/spelling/collocation payloads so a learner typing `ran` for `run` grades
correct — a form-set widening at grade time, NOT a `match_key` change.

**Targeted example augmentation**: `add_examples(sense_id, n=3, theme=None)` is the
ONE additive-generation surface — an example is an open-ended illustration of a
sense, so authoring more never fabricates a linguistic fact (unlike senses or
collocations, which are finite/attested and NOT augmentable). It APPENDS up to `n`
fresh tagged examples to one sense (best-effort max, never overwrites; `n` clamped
to the schema ceiling), feeding existing examples to the prompt for soft dedup, and
returns the updated `SenseView`. Embeddings are untouched (they cover the definition
only). `theme=` augments the sense's themed overlay instead — the overlay must
already exist (word themed via `generate(theme=)`), else `ValueError`; it never
silently themes the whole word. Both neutral and themed examples carry the same
`<t inf>` markup contract, so the whole-word `themed_restyling` path emits tags too.

**Topic tags** (open-vocabulary, LLM-authored): each generated word gets 1-3 tags
emitted in the same LLM call as senses. Consistency without embeddings: the full
existing tag vocab is injected into every generation prompt for reuse, a
deterministic `tag_key` (lowercase / singular / diacritic- and control-folded)
dedups case/plural variants on the write path, and resolve-or-create under the
UNIQUE key keeps one row per tag (title set once, first-seen). Browse via
`list_tags()` (live member count) and `list_entries_by_tag(tag)` (exact filter,
resolved through `tag_key`). Both FREE — 0 LLM calls. Curation
(`rename_tag`/`delete_tag`/`merge_tags`) edits/removes/folds tags after the fact —
`tag_key` stays immutable (only display name/title change on rename; a merge
re-points `word_tags` from the source tags onto the destination and deletes the
sources).

## Managing & batch APIs

Every resource gets get/list/delete alongside its create path: `delete_entry` +
`list_entries` (paginated dictionary browse, `SearchResult` rows) alongside
`generate`/`get_entry`; `get_theme`/`update_theme`/`delete_theme` alongside
`create_theme`/`list_themes`; `rename_tag`/`delete_tag`/`merge_tags` alongside
`list_tags`; `get_asset`/`list_assets`/`delete_asset`/`purge_assets` alongside
`translate`/`tts`. Bulk variants (`generate_many`, `get_many`, `translate_many`,
`get_status_many`, `QuestionEngine.grade_many`) fan out concurrently (bounded by
a `concurrency` semaphore where an LLM may be involved) over the SAME single-item
method, so existing guarantees (per-`match_key` lock, cache-first) apply
unchanged. Every batch call returns `list[BatchResult]`, order-aligned with the
input: one item's failure is captured in that slot's `error` and never aborts the
rest. `stats()` is a read-only counter (no LLM, one round of grouped COUNTs in a
single session): words-by-status, senses, examples, tags, themes, themed-words
(distinct words with ≥1 overlay), assets-by-kind, and questions.

## Question engine

The question services turn a `done` entry into vocabulary questions and grade
answers. They *manage* questions (create / read / delete / grade); they do not
*use* them — rotation, quiz sessions, SRS, and progress are the application's job.
Two engines exist per process: the reader's, built without providers, and the
worker's, built with the LLM, the rubric judge, and the speech port.

**Three axes wired through `answer_kind`.** A **format** declares an `answer_kind`
(what an answer looks like: `single_choice` / `text_span` / `free_text` /
`matching`); a
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

The format set proves the abstraction by covering every backend combination:

| Format | answer_kind | Generator | Grader | Persists |
|--------|-------------|-----------|--------|----------|
| `definition_mcq` | `single_choice` | rule | rule (index) | no |
| `pronunciation_mcq` | `single_choice` | rule (IPA) | rule (index) | no |
| `cloze` | `text_span` | rule | rule (`match_key`) | no |
| `collocation_fill` | `text_span` | rule | rule (`match_key`) | no |
| `contextual_mcq` | `single_choice` | llm | rule (index) | yes |
| `use_in_sentence` | `free_text` | rule | llm (rubric) | no |
| `matching` | `matching` | rule | rule (permutation) | no |
| `listening` | `single_choice` | rule (TTS) | rule (index) | yes |
| `spelling` | `text_span` | rule (TTS) | rule (`match_key`) | no |

`pronunciation_mcq` and `collocation_fill` reuse already-stored content (per-POS
IPA, and the `collocations` child table) — no new generation cost. `contextual_mcq`
(llm-generated, rule-graded) and `use_in_sentence` (rule-generated,
llm-graded) are the cross-axis proofs. `listening`/`spelling` synthesize an audio
clip via the configured TTS provider (addressed by the durable
`(source_kind, source_id, voice, fmt)` reference tuple, not a row id, so the payload
survives a purge/regenerate); with no TTS configured they degrade to no questions
rather than failing. `cloze`/`collocation_fill`/`spelling` grade `text_span` by
`match_key` equality against the answer PLUS the sense's `accepted_forms` (its
inflected surfaces), so a learner typing `ran` for `run` scores right without
touching the `match_key` invariant; `cloze` also blanks the exact target span from
the example's `<t inf>` markup (via `lexi_ai.markup`), falling back to a
word-boundary match when a span is absent. `pronunciation_mcq` reuses the MCQ
machinery with the sense's IPA as the stem; `collocation_fill` blanks the target in
a stored collocation. Adding a format is one plugin **module** + one
`register(...)` line: each format lives in its own file under
`lexi_ai/questions/formats/` (cross-format helpers in `_shared.py`) and
self-registers on import, so the package `__init__` importing it populates the
registry. The registry validates the format↔answer_kind coupling at
import time (a mis-wire is an import error). Payload is app-level JSON in a `Text`
column (the one deviation from native typing), (de)serialized only at the
repository boundary, which rejects an embedded NUL so it round-trips safely on
Postgres. Distractors are best-effort (semantic neighbours, then shared topic
tags); an MCQ degrades to fewer options rather than fabricating. The LLM plugin and
judge are injectable, so the whole subsystem tests with fake runnables and zero
network.

**Portability:** only `Text`/`String`/`Integer`/`DateTime` — no
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
  `cambridge_cefr` map (built by the generation service from the bundle) — no
  source coupling.
- **Stub rows** (#11): a mentioned related word becomes a `pending` `words` row
  immediately, so links are real ids and the stub doubles as the lazy-gen queue.
- **Async safety:** the repository never touches relationship collections on a
  *persistent* object (that triggers a lazy-load outside greenlet context);
  children are cleared with Core `delete()` and re-inserted with explicit FK ids.
  Read models are built inside the session via `selectinload`.

## Sense vectors live outside the primary database

Semantic search ranks in the vector index and hydrates from SQL. The index owns
which senses are embedded; the relational store answers only "which done senses
exist". One source of truth, and the row shape of the two stores never has to
agree on anything but the sense id.

```
semantic_search(q) → embed(q) → VectorIndex.query(v, k+overfetch, {model})
                             → SenseRepo.semantic_rows(ids)  → SemanticHit[]
```

**Eventually consistent, deliberately.** A vector cannot join the SQL transaction
that publishes an entry, and embedding was already a post-commit best-effort step,
so keeping the BLOB in `senses` only pretended the two were atomic. The contract is
now explicit: a missing or stale vector is a tolerated transient, never a failure.

- **Missing** — generation embedded nothing (extra not installed, encoder OOM). The
  entry is still `done`; `backfill_embeddings` fills the gap.
- **Stale model** — vectors carry the encoder's model name in metadata and queries
  pre-filter on it, so changing encoders yields no hits (not wrong hits) until the
  backfill re-embeds. Re-embedding replaces the vector in place, keyed by sense id.
- **Orphaned** — a delete or a regeneration leaves a vector whose sense is gone.
  Hydration skips unknown ids so an orphan can never surface as a result;
  `delete_entry` forgets vectors immediately, and the backfill prunes the rest so
  orphans cannot crowd real hits out of a top-k.

Failures surface; only the generation hook swallows them. `semantic_search` and
`backfill_embeddings` RAISE when the encoder or index is broken — an empty result
means "nothing matched" and a `0` means "nothing needed doing", so neither can be
confused with a broken installation. The single exception is the post-commit
embed hook (`Lexicon._embed_words`): the entry is persisted and the LLM call is
already paid for, so a vector failure there is swallowed and the backfill
reconciles it later.

**The feature is opt-in and off by default.** `LEXI_VECTOR_BACKEND=none` (the
default) makes `build_vector_index` return `None`, and the services that would use
an index say so: `semantic_search` and `backfill_embeddings` raise
`SemanticSearchDisabled`, while the embed hook and `delete_entry` skip silently
because neither asked for the feature. `None` rather than a null object precisely
because those two groups want opposite behaviour, which one object cannot express.
Every reason the feature can be unavailable — disabled, backend extra missing,
encoder extra missing — is a `SemanticSearchUnavailable`, so a caller degrades with
one `except` instead of enumerating causes it will inevitably under-count.

**Backends** are a settings switch (`LEXI_VECTOR_BACKEND`), not a code change:
`lancedb` (embedded, on disk, ANN, needs the `[lancedb]` extra — the choice for
production) or `memory` (exact scan, non-durable, what the hermetic test tier
uses). A backend whose extra is absent fails at construction, not at first query:
selecting it is an explicit opt-in, so that is where the error belongs. Adding
pgvector or Qdrant is one module in `infrastructure/vectors/` plus a branch in
`build_vector_index`; nothing upstream of the port changes. The adapter contract is
pinned by one test module parametrized over every backend, with the exact-scan
in-memory index as the ground truth.

## Adopted embedding boundary

`lexi_ai` is the dictionary domain library. It owns transport-agnostic models,
repositories, generation, questions, grading, translation, and TTS; consumers
build one `Lexicon` and take the `LexiconReader` or `LexiconEngine` facade that
matches what that process is permitted to do.
It must not own HTTP/gRPC adapters, Redis queues, task/outbox orchestration, or
service deployment credentials.

Pycil uses the library as its only Lexi execution dependency. It runs against a
single PostgreSQL cluster with Lexi tables isolated in the `lexi` schema and
separate Alembic versioning. Pycil owns commands, global task identity, durable
task interests, outbox delivery, and provider-worker execution.

The companion contract is
[Pycil's SSE task orchestration ADR](../../pycil/docs/decisions/sse-task-orchestration-boundary.md).
Its task/outbox lifecycle remains outside `lexi_ai`, which keeps importing the
library free of transport and worker dependencies.

## Configuration

Env vars (prefix `LEXI_`): `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`,
`LLM_TEMPERATURE`, `DB_URL` (generated DB), `CAMBRIDGE_DB_PATH` (read-only source).
Assets/themes: `ASSET_CACHE_DIR` (TTS clip dir, default `./lexi-assets`),
`TRANSLATE_MODEL` (optional translation model override, falls back to `LLM_MODEL`),
and `TTS_BASE_URL`/`TTS_API_KEY`/`TTS_MODEL`/`TTS_VOICE`/`TTS_FORMAT` for the
OpenAI-compatible TTS provider. When a `TTS_API_KEY` is set, `TTS_BASE_URL` must be
`https://` (or a loopback host) or construction raises — the key is never sent in
cleartext. With none configured, TTS falls back to the stub (raises, never caches
fake audio).

Sense vectors: `VECTOR_BACKEND` (`lancedb` default, or `memory`), `VECTOR_PATH`
(LanceDB store dir, default `./lexi-vectors`), `VECTOR_METRIC` (`cosine` — must
match the encoder's geometry; the Embedder L2-normalizes). Encoder knobs stay
`EMBEDDING_MODEL`/`EMBEDDING_DEVICE`/`EMBEDDING_BATCH_SIZE`/`EMBEDDING_MAX_LENGTH`.

**Schema versioning:** the ORM is the single schema source. SQLite local
development uses `init_models` to create a fresh disposable database; PostgreSQL
deployments run the Alembic chain, which owns the `lexi` schema and
`lexi.alembic_version`, and runtime processes never perform DDL. The chain is one
autogenerated baseline, and `alembic check` fails if the models and that baseline
ever diverge. The baseline assumes an empty database, and later changes are
ordinary forward revisions on top of it.

## Persistence boundaries

Each aggregate has its own repository behind a Protocol in `domain/ports.py`, and
a `SqlAlchemyUnitOfWork` hands them the one session they share, so several calls
compose into a single transaction. Three boundaries are load-bearing, and folding
any of them into the wrong transaction fails quietly rather than loudly:

- **The generation claim commits before provider work.** Competing workers detect
  ownership by reading the epoch, so an uncommitted claim is invisible and the
  fence stops fencing — both workers proceed and one overwrites the other.
- **Enrichment stays outside the publish.** Embedding and relation resolution are
  best-effort and run after the commit. Inside it, an embedding failure would roll
  back published content; the relation queue also reads `status = "done"`, so
  pre-commit it would find nothing. Embedding could not be inside it in any case —
  vectors live in a separate store that cannot join a SQL transaction.
- **Error recording uses an independent session.** It runs after the publish
  transaction already rolled back, and a rolled-back session cannot write.

Assets and questions are deliberately not on the unit of work: the asset cache is
reconciled after the write commits, and questions are written by the assessment
plugins in their own transaction. Both would otherwise put a best-effort step
inside a transaction that must not roll back because of it.

Controlled vocabularies are enforced by the column type (`infrastructure/db/
types.py`), not only by the LLM output schema, so every writer is covered. A
drifted label is quiet damage: the relation part-of-speech filter compares labels,
so a stored `adj` where `adjective` is expected mis-filters candidates silently.

`db.py` corrects two SQLite driver defaults. Foreign keys need a per-connection
pragma, and pysqlite's implicit `BEGIN` breaks SAVEPOINT — without the fix a write
made inside `begin_nested()` (the concurrent-unique-key recovery path) survives a
rollback, so a failed multi-step write leaks committed stub rows.

## Testing

507 tests, `uv run pytest`. No live LLM calls (fake runnables). Reference and
phrase-overlap tests run against the real Cambridge `./data` and skip if absent.
Themes and cached assets add `tests/test_themes.py` (theme_key folding, dedup,
schema compile), `tests/test_theming.py` (themed generation + read overlay with a
fake generator, count-mismatch guard, per-sense fallback, concurrent overlay
single-flight), and `tests/test_assets.py` (reference addressing with read-time
`content_hash` verify, param normalization, TTS allow-list validation, resolve-or-
create, translation cache hits, TTS stub-raises + real-provider round-trip).

The suite includes regression tests for a multi-agent review pass: Postgres
NUL-safety of `match_key`, concurrent-insert savepoint recovery, the
Cambridge-first CEFR contract (`sense#42` vs bare `42`), per-key lock eviction,
and the word-enrichment write→read spine (word-reference normalization/dedup,
closed-vocab enum rejection, collocation ordering + sanitization, eager-load).
The question engine is covered end-to-end: each format's generate + grade, the
two cross-axis proofs, plugin-owned persistence (the engine stays blind to it),
the registry coupling guard, a one-line new-format extensibility proof, and
payload round-trip (unicode + NUL rejection).

### PostgreSQL tier (opt-in, `LEXI_TEST_PG_URL`)

An opt-in Postgres tier (`tests/test_postgres_integration.py`) exercises the
defect class invisible to SQLite: NUL rejection, `VARCHAR(n)` length enforcement,
and tz-aware datetime binding on `TIMESTAMP WITHOUT TIME ZONE`. asyncpg is strict
where SQLite is lax; the SQLite-only CI could not catch these.

The tier requires the `asyncpg` driver and a disposable Postgres database:

```
uv sync --extra postgres
LEXI_TEST_PG_URL=postgresql+asyncpg://user:pass@localhost/lexi_test uv run pytest
```

Phase 4 regressions cover **all five untrusted write columns** (`norm`,
`alias_norm`, `source_ref`, `definition`, `example`) and the H2 tz-aware datetime
bind that caused an unbounded WSD re-judge loop. Each regression was confirmed to
FAIL against pre-fix code (Phase 0 red-before-green discipline) and passes after.

The tier skips cleanly when the driver or URL is absent — the default
`uv run pytest` stays hermetic and network-free.
