# Lexi-AI — Implementation Notes

> Living note: things that should belong in lexi-ai but are not implemented yet.
> External consumers (e.g. Pycil) depend on these and treat them as future
> capabilities of this library, not of the consumer's own worker.

## 2026-07-05 — TTS audio + definition translation belong in lexi-ai

**Decision:** Two capabilities currently live in the Pycil worker
(`apps/worker/worker/jobs/tts_render.py`, `translate_definition.py`) should move
**into lexi-ai** as first-class library features. Pycil's MVP defers TTS +
translation entirely until lexi-ai ships them, rather than carrying its own
implementations.

### What to implement in lexi-ai (future)

1. **TTS audio render** — text → speech. Likely a thin client over an external
   TTS provider (ElevenLabs / Google TTS / OpenAI TTS) with a cached artifact
   URL persisted in the sense/definition row. Surface as e.g.
   `await lex.tts.render(text, lang="en") -> TtsArtifact` returning a URL +
   provider metadata. Keep the provider pluggable and model/endpoint env-driven
   (`LEXI_TTS_*`), matching the existing `LEXI_LLM_*` config pattern.

2. **Definition translation** — translate an English sense definition into the
   learner's L1. LLM-driven (the `LEXI_LLM_*` client already in place). Persist
   per `(sense_id, target_lang)` so repeats are free, mirroring the lazy-lookup
   philosophy. Surface as e.g.
   `await lex.translate(definition, target_lang="vi") -> Translation` keyed by
   match-key so case/variant folding reuses the same row.

### Design constraints to honor

- **Portable storage** — the one-schema-runs-on-SQLite-and-Postgres rule
  (portable column types only, no JSONB/ARRAY/native enum) must still hold.
  TTS/translation columns/tables use plain types.
- **Lazy + cached** — both must follow the "first lookup spends tokens / API
  calls, every lookup after is a free cache hit" contract. No eager generation.
- **No hardcoded provider/model** — env-driven, same as `LEXI_LLM_MODEL`.
- **Optional extra** — consider gating heavy TTS deps behind an optional extra
  group (like `embeddings`), so the core library stays light when unused.

### Why here, not in Pycil

Pycil's domain is the learning loop (hierarchy, FSRS scheduling, sessions).
TTS + translation are dictionary-content concerns — they enrich the `Entry`
that lexi-ai already owns. Keeping them in lexi-ai means every consumer of the
library gets them, and Pycil's worker stays a thin dispatcher over
`lex.generate / lex.questions.generate / lex.questions.grade / lex.tts.render /
lex.translate`.

### Pycil impact (consumer side)

Pycil's worker will retain a `tts_render` job and a `translate_definition` job
in its queue interface, but their bodies collapse to a single call into lexi-ai
once these land. Until then, Pycil ships **without** TTS + translation.

## 2026-07-05 — Add `get_senses(sense_ids)` batch lookup to the public API

**Gap:** the public `Lexicon` API has no way to fetch senses by their DB ids.
`Lexicon.get(lexi_word_id)` (`api.py:211`) takes a **word** id and returns the
whole `Entry` (all senses of that word). Consumers that track the atomic unit as
`sense_id` (Pycil keys its per-user FSRS stats + deck membership on `sense_id`)
have no lookup path — they hold sense ids but can only ask for word entries.

**To implement:**
`async def get_senses(self, sense_ids: list[int]) -> list[SenseView]` on
`Lexicon`, backed by a repository query that selects `Sense` rows by id
(+ their examples/collocations/references) and assembles `SenseView`
(definition, cefr_level, pos, guideword, examples, ...) — the same assembly
`_build_entry` already does per sense (`api.py:~532`), factored to run from a
sense-id list instead of a word. Batch (one query for N ids), preserve input
order or return a dict keyed by sense_id.

**Why here, not in Pycil:** Pycil must not JOIN or FK into lexi-ai's tables
(separate database, opaque `sense_id` refs — see Pycil's dictionary-ownership
boundary). The only clean way to resolve sense content is a library/RPC method
lexi-ai owns. This is also a natural future gRPC method.

**Pycil dependency:** Pycil's `LexiClient` port declares `get_senses(sense_ids)`
and its hierarchy/stats + deck-item endpoints call it to resolve CEFR + display
content. Pycil's plan (phase-05) has a step to add this to lexi-ai first.

## 2026-07-09 — Sense-level relations: new public read field (NON-breaking)

**Change (additive):** semantic relations (`synonym`, `antonym`, `hypernym`,
`hyponym`, `meronym`, `holonym`, `see_also`) moved from the word level to the
**sense** level. They are now emitted per sense with a target gloss, persisted as
`sense_relation` half-edges, and reconciled to a specific target sense by an
LLM-judge WSD pass. A new public read field surfaces them:

- **New:** `SenseView.relations: list[SenseRelationView]` (default `[]`). Each
  `SenseRelationView` carries:
  - `rel_type: str`
  - `to_word_display: str`, `to_word_id: int`, `to_word_status: str` — always
    present (the edge is at least sense→word).
  - `to_sense_id: int | None` — set once WSD resolves it (else `None`).
  - `to_sense_gloss: str | None` — the resolved target sense's definition.
  - `wsd_state: str` — DERIVED (`'resolved' | 'unresolvable' | 'pending'`), NOT a
    DB column. Consumers filter on it. A resolved edge whose `target_hash` no
    longer matches the target sense's current definition is surfaced as
    `pending` (hash-verified on read).

**NOT changed (no consumer breakage):**
- `Entry.links` / `LinkView` shape is unchanged. WORD-level relations
  (`word_family`, `confused_with`, `variant_of`, `arrow_redirect`,
  `another_word`, `part_of_phrasal_family`) still surface there exactly as
  before — only the underlying table renamed (`entry_links` → `word_relation`),
  invisible to consumers.
- `SenseView` is constructed by keyword, and `relations` is a defaulted field, so
  positional/older consumers are unaffected.

**New public method:** `await lex.resolve_relations(batch_size=20)` runs one
manual/backfill WSD batch (returns `list[BatchResult]` per edge). Automatic
reconciliation also fires as a best-effort hook whenever a target word is
generated, so coverage grows with traffic without a scheduler. Requires an LLM to
be configured (degrades to a no-op otherwise).

**Migration:** `init_models` is additive `create_all`; a one-time
`migrate_relations()` (in `db.py`, wired into `init_models`) renames the legacy
`entry_links` table to `word_relation` before `create_all` so existing DBs keep
their word-level data. `sense_relation` is created fresh.
