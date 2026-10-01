# Lexi AI — Technical Design

**Audience:** CTO, business analysts, QA/test engineers, and integration owners<br>
**Status:** Design of the current product boundary; not a roadmap or an implementation guide<br>
**Scope:** The active `lexi_ai` library and its `Lexicon` facade; excludes archived v1

## 1. Executive summary

Lexi AI is an on-demand English learner's dictionary and exercise engine delivered as an asynchronous library to a trusted consumer application. A consumer can search existing dictionary content, discover eligible source entries, explicitly generate a selected entry, read its meanings, optionally create an alternate writing Theme, generate exercises for a meaning, and grade answers. It can also resolve semantic relations between generated meanings and translate selected text.

The system has two distinct stores: an **immutable reference snapshot** of Cambridge entries and a **writable generated dictionary**. Cambridge establishes which lexical item the user selected; generation synthesizes learner-oriented material from that entry, with optional WordNet supporting evidence. Provider output is validated before publication. Generated content, theme variants, exercises, and translations live in the writable store. A separate decision capability supports semantic grading and Sense Linking.

Lexi AI does **not** choose what to generate on its own, run a background scheduler, expose an HTTP API, manage learners or mastery, or decide when correct answers may be revealed. Those responsibilities belong to the consuming product.

## 2. Goals, boundaries, and actors

### Goals

1. Generate one stable dictionary **Word** from one explicitly selected, eligible source entry.
2. Represent multiple distinct **Senses** of that Word, each with one learner definition per namespace, examples, lexical metadata, and optional relations.
3. Support exact neutral and themed content namespaces without changing the meaning inventory.
4. Produce reusable exercise artifacts that support the response formats appropriate to each question type.
5. Return a small grading verdict, leaving presentation and learner progress to the consumer.
6. Maintain a traceable, read-only source boundary and publish generated content atomically.

### Actors and ownership

| Actor / system | Responsibility | Does not own |
| --- | --- | --- |
| Trusted consumer application | User experience, selecting a source entry, sequencing operations, overlapping-request coordination, authorization, learner-safe question presentation, answer reveal, progress/mastery, operational retries | Dictionary generation rules or semantic verdict generation |
| Lexi AI | Lexical search, selected-entry generation, dictionary and exercise persistence, grading, explicit Sense Linking, Theme and translation operations | HTTP endpoints, scheduling, user identities, progress tracking |
| Cambridge snapshot | Read-only source entries, meanings, examples, pronunciations, and source identifiers | Generated content or learner state |
| Optional WordNet corpus | Supporting generation evidence when a citation is suitable and the corpus is available | Choosing the requested entry |
| Structured language-model provider | Produces structured words, themed text, questions, theme descriptors, and translations | Publishing unvalidated output directly |
| Decision provider | Evaluates free-text answers and selects target senses for relations; an LLM-backed decision route may handle low-confidence applicable choices | Learner-facing presentation or an unconditional fallback for outages |
| Generated dictionary database | Durable Words, Senses, content, relations, themes, questions, and translation cache | Source-snapshot mutation |

### Explicit exclusions

No automatic corpus ingestion, free-form generation without a selected source entry, HTTP server, authentication or tenancy model, background Sense Linking, spaced repetition, learner analytics, audio/media pipeline, or frontend is specified by this design. An integrating product must provide those capabilities separately if needed.

## 3. Product capabilities and user journeys

### 3.1 Discover and generate a Word

1. The consumer submits a lexical query. Results contain two separate groups: **generated Words** already available to read and, if requested, **available source entries** that can be selected for generation.
2. Each available entry has an opaque selection handle, display label, and entry type. Distinct eligible source entries remain distinct even if their labels match. Entries already consumed by completed Words are omitted from the available group.
3. The consumer selects exactly one handle. If that source entry has already been generated, the completed Word is reused rather than regenerated. Otherwise, the system reads that entry and optionally supporting WordNet evidence, requests one structured Word, checks the result against source identity and citations, and publishes the complete Word in one operation.
4. A Word may contain several Senses; those Senses must belong to the **same selected lexical item**. A selected page cannot silently produce multiple Words. Related lexical items may be represented as pending relation targets rather than generated dictionary entries.
5. The consumer receives a detached Word view. The source snapshot remains unchanged.

**Search behavior:** Generated results prioritize exact lemma, exact alias, exact form, licensed expression pattern, prefix, then approximate spelling suggestions across lemmas, aliases, and Sense-owned forms. A matched surface and match category explain why a Word appeared; no public similarity score is added. Pattern matching must match the whole expression and may substitute only licensed inflected head forms belonging to the relevant Sense. Search returns at most 30 distinct ranked Words, choosing the best match per Word before applying the result limit. This limit does not restrict eligibility to the earliest entries in the dictionary. Queries of one or two normalized characters use exact/prefix retrieval only. Approximate spelling suggestions remain distinct from exact lexical matches and do not establish semantic equivalence.

**Scale boundary:** PostgreSQL is the production search backend. Three native GIN trigram indexes filter approximate candidates at similarity threshold 0.3; the database performs similarity ordering and per-Word deduplication. Equality/prefix retrieval uses normalized keys and B-tree indexes. There is no Python fuzzy engine, raw-candidate cap, external search service, or mirrored search projection. Pattern matching remains a separate operation. The database may choose different plans based on selectivity; returning 30 Words does not mean processing only 30 candidates. SQLite is a lexical-only development/test backend, not an equivalent fuzzy implementation. Trigram similarity can miss short-word typos or prefer a shorter related spelling over the intended correction.

**Selection boundary:** An available handle identifies one source entry; it is not a public source ID, a search query, or permission to generate another entry. Missing or ineligible source evidence must fail without publishing a partial Word.

### 3.2 Read a dictionary Word

A completed Word contains its citation lemma, entry type, aliases, Word-level related items, and Senses in ID order. Each Sense carries its part of speech, learning-priority tier, one definition, tagged usage examples, inflected forms, licensed patterns, collocations, optional CEFR level/register/usage note/pronunciations, and Sense-level semantic relations. The consumer can read one Word or request multiple Senses by ID. Missing or unpublished Words do not appear as completed dictionary entries.

Each Sense/namespace has exactly one Definition, exposed as `definition`; only example counts are configurable. Definition and Example remain relational. Sense has no position field; dense positions apply only to Question retrieval. The learning-priority tier is not a synonym for “first” or “default” meaning. Example text identifies actual target surfaces with a tagged inflection so the consuming product can highlight the correct span without guessing.

### 3.3 Manage writing Themes

A Theme has a stable key, display name, voice, and diction. A concept can be turned into voice/diction descriptors; metadata can later be listed, read, updated, or deleted.

The neutral Word and its Sense inventory are generated first. When a Theme is requested, the system creates a **complete themed set of definitions and examples for every existing Sense**. Theme changes expression and example context, not identity or meaning. A themed read uses only content in that Theme namespace: it does not silently fall back to neutral definitions/examples. If a complete themed set does not exist, the themed Word is unavailable until generated. Repeat requests reuse an already complete set. Updating Theme metadata does not retroactively rewrite saved themed content; new questions use the current Theme descriptors. Deleting a Theme removes its themed content and themed questions while retaining neutral material.

### 3.4 Generate, present, and grade questions

The consumer explicitly chooses a Sense, question type, number of questions, distractor count, and optional Theme. Generation uses the requested content namespace; themed exercises require the complete themed Word. One structured generation request creates a validated batch, and the batch is saved as a single unit. Repeating the operation creates **new** artifacts; reading or retrieving an existing question does not regenerate it.

Every saved Question is a **trusted-server artifact**: it contains the prompt, correct option, wrong options, option explanations, and, where applicable, acceptable stored alternatives. Option IDs are stable within the saved artifact. A question can be retrieved by ID, listed by Sense and exact Theme namespace, randomly retrieved from that scope, or deleted. The consuming server must build a learner-safe view that omits the answer and explanations until disclosure is appropriate.

| Question type | Learner task | Supported answer formats |
| --- | --- | --- |
| `definition_to_word` | Identify the Word from one displayed definition | `single_choice`, `single_word` |
| `context_to_word` | Supply the fitting expression from a situation | `single_choice`, `single_word` |
| `cloze_to_word` | Fill one complete-answer blank in context | `single_choice`, `single_word` |
| `word_to_definition` | Identify or describe the Word's meaning | `single_choice`, `short_answer` |
| `word_to_usage` | Identify or produce correct usage for a displayed meaning | `single_choice`, `short_answer` |
| `dialogue_completion` | Complete a named-speaker dialogue with one missing reply | `single_choice` |
| `meaning_in_context` | Interpret a tagged expression in a passage | `single_choice` |

These are **seven question types and twelve allowed type/format combinations**. A single saved artifact can be used with either allowed format; no parallel format-specific artifact is required. Question creation requests 1–20 artifacts with 3–20 distractors each. Validity includes the requested batch size, nonempty and distinct options, one complete cloze blank where relevant, a single missing dialogue reply, tagged target surfaces where the task displays the target, and preservation of fixed Word/definition anchors where required.

For `definition_to_word` and `context_to_word`, the application attaches the stored Word's citation lemma as the correct answer. For `word_to_definition`, it attaches the single stored definition from the requested namespace. Those model output schemas contain no `correct` field; the model supplies an explanation and distractors, plus the required question content. Other types retain model-authored correct options when the answer depends on new content.

`dialogue_completion.target_placement` is saved with the artifact. The default `dialogue` placement marks the target in a visible turn; `options` hides it in the visible conversation and requires the complete tagged target in every option. Invalid placement or target markup is rejected before publication.

**Grading contract:**

- `single_choice` compares a submitted saved option ID with the artifact's options; it does not require a decision provider. Unknown option IDs are invalid submissions, not simply incorrect answers.
- Choice and single-word grades return independent `task_fit`/`spelling_error` booleans plus `sense_id`. Exact saved single-word answers accept case/Unicode/spacing normalization without inference. Other answers first judge task fit and spelling; only a fit answer without spelling error searches the top-ranked Word and selects from its complete Sense inventory. Missing mapping never changes correctness.
- Definition short answers first identify the intended Sense from all Senses of the owner Word, even when the response is imperfect. If none is identifiable, `sense_id`, `accuracy`, and `coverage` are null; otherwise a second stage returns `accuracy=accurate|mixed|inaccurate` and `coverage=sufficient|partial|minimal`.
- Usage short answers gate on `used`. If false, diagnostics are null; otherwise a second stage independently returns `meaning`, `form`, `construction`, `collocation`, and `appropriacy` against the saved meaning anchor.
- These diagnostic values are not mastery scores, learner attempt records, explanations, or instructions to reveal an answer. The seven decision templates execute as JSON-e at runtime, not inline rubrics.

### 3.5 Resolve semantic relations

Generation may create Word-level links (word family, easily confused terms, or source-derived phrasal-family links) and Sense-level links (synonym, antonym, hypernym, hyponym, meronym, holonym). A Sense-level relation initially points to a **target Word** plus a target-meaning gloss; it does not assume which target Sense is intended. Relation targets may exist only as pending lexical placeholders until separately generated.

Sense Linking is an **explicit consumer-triggered operation**, not a side effect of Word generation. Eligible generated targets are considered in bounded edge batches, with every same-POS target Sense included, never a truncated inventory. A candidate must match both the target gloss and the directional relation rule with the source meaning; incomplete neutral evidence and provider-size failures are explicit per-edge errors. The decision capability can select one candidate or none. The observable `resolution_state` is `pending`, `resolved`, or `unresolvable`. A resolved link is tied to a fingerprint of its target definition. Changes to source neutral meaning, target neutral definitions, or the relevant Sense inventory return affected decisions to pending, including prior no-match decisions; themed content does not. Unchanged decisions need not be reassessed. Evidence must still be current when a decision is saved. Independent links may be resolved concurrently, and one link's provider failure must not erase successful results for other links.

### 3.6 Translate selected text

The consumer supplies selected text and a target-language code. Target-expression markup is validated and unwrapped before translation or cache lookup; malformed tags are errors, not translatable prose. The system caches by the **exact unwrapped text sent to the provider** plus target language. Tagged and plain versions of the same text share a key; whitespace is preserved, and changing whitespace or language makes a distinct key. A cache hit needs no language-model call. Translation records can be listed, retrieved, individually deleted, or purged. This is a text cache, not automatic translation of all Word fields, and it does not attach translations to source rows or learner accounts.

## 4. Conceptual information model

```text
Reference entry (read-only) --selected by handle--> Word (generated)
                                                |-- aliases
                                                |-- source association
                                                |-- Word relations --> target Word
                                                `-- Senses (ordered)
                                                     |-- one definition and examples (neutral or Theme)
                                                     |-- forms, patterns, collocations
                                                     |-- source references
                                                     |-- Sense relations --> target Word
                                                     |                       `--> optional target Sense
                                                     `-- Questions (neutral or Theme)

Theme --> themed definitions/examples and Questions
Exact input text + target language --> cached Translation
```

| Concept | Identity and key design rule | Lifecycle / ownership |
| --- | --- | --- |
| Reference entry | Opaque available handle resolves to one eligible source entry | External, read-only; not copied wholesale |
| Word | One normalized citation identity; one selected-source association for a generated entry | Pending placeholders can exist as relation targets; only completed Words are public dictionary entries |
| Sense | Ordered meaning within one Word, with one part of speech and tier | Shares its identity across neutral and themed presentations |
| Definition / Example | One Definition and multiple Examples per Sense/namespace; neutral or one Theme | Deleted with its Sense or Theme, as appropriate |
| Source reference | Cites a supplied Cambridge Sense or optional supplied WordNet synset | Validated during generation; provenance, not an alternate Word identity |
| Word relation | Typed link to another Word | May point to a pending target Word |
| Sense relation | Typed link from a Sense to a target Word, optionally refined to a target Sense | Separate Sense Linking lifecycle with freshness tied to target definitions |
| Theme | Reusable metadata and content namespace | Deletion removes its themed material and questions, not neutral content |
| Question | Saved full answer-bearing exercise tied to a Sense and one namespace | Append-only generation; individually retrievable and deletable |
| Translation | Exact-text fingerprint plus target language | Cache entry independent of Word/Sense identity |

**Identity discipline:** Citation identity normalizes Unicode compatibility, casing, and spacing while preserving meaningful lexical differences such as fixed particles, hyphens, and literal words versus typed variable slots. An alias is searchable but does not create a second Word. Licensed variable slots in expressions are explicit and bounded; source alternatives or ambiguous optional notation must not be silently collapsed into a different canonical lemma.

**Deletion discipline:** Removing a parent removes dependent content in its own scope. Removing a Theme must not relabel themed content as neutral or remove the underlying Word. The reference snapshot is never a deletion target of generated-dictionary administration.

## 5. End-to-end sequences and consistency

### Sequence A — selected-entry publication

```text
Consumer -> Lexi AI: search(query, include available entries)
Lexi AI -> generated dictionary: find completed matches
Lexi AI -> reference snapshot: find eligible unconsumed source entries
Lexi AI -> Consumer: generated hits + optional available handles
Consumer -> Lexi AI: generate(selected handle, optional Theme)
Lexi AI -> generated dictionary: reuse if this source is already complete
Lexi AI -> reference snapshot / optional WordNet: obtain evidence
Lexi AI -> structured provider: request exactly one Word
Lexi AI: validate identity, citations, markup, counts, and vocabulary
Lexi AI -> generated dictionary: publish complete neutral Word atomically
Lexi AI -> generated dictionary/provider: reuse or create complete Theme set if requested
Lexi AI -> Consumer: detached Word in the requested namespace
```

### Sequence B — practice and evaluation

```text
Consumer -> Lexi AI: generate N questions for one Sense/type/namespace
Lexi AI -> generated dictionary: read the requested complete content set
Lexi AI -> structured provider: produce N questions and K distractors each
Lexi AI: validate the complete batch
Lexi AI -> generated dictionary: append the whole batch atomically
Lexi AI -> Consumer server: full answer-bearing artifacts
Consumer server -> Learner: redacted prompts/options only
Learner -> Consumer server: response
Consumer server -> Lexi AI: grade(saved question ID, allowed format, answer)
Lexi AI -> Consumer server: staged task-specific diagnostics
Consumer server: own feedback, answer disclosure, attempt history, and mastery
```

### Consistency rules

- Operations are consumer-initiated and scoped to short transactions; long-running provider work is performed before publication, not as a partially visible Word or question batch.
- Complete neutral publication, complete themed content for a Word, and a newly generated question batch are all-or-nothing within their respective operations. A request with both neutral and Theme generation has two stages: neutral completion may remain available if the later themed stage fails.
- Reuse is defined for a previously consumed selected entry and a previously complete Word/Theme set. Question generation is intentionally additive; translation is intentionally cached by exact text and language.
- The consumer must serialize or otherwise coordinate overlapping writes for the same selected Word, Theme set, or translation cache key. This library is not a distributed job coordinator.
- Decision outcomes use one configurable, inclusive confidence threshold for applicable decisions. Low-confidence applicable choices may use a separately configured decision route; provider transport failure is not itself a semantic “no match.”

## 6. Trust, safety, and privacy boundaries

1. **Answer confidentiality:** Saved Questions carry correct answers and explanations. Only a trusted server should receive the complete artifact; a learner-facing payload must be explicitly redacted. The library does not implement this redaction or user authorization.
2. **Untrusted inputs:** Reference text, user text, Theme concepts, and generated-dictionary content are data, not instructions. Structured tasks are separated from that data, and published provider responses must satisfy constrained contracts and domain checks.
3. **Source integrity:** Source entry access is read-only. Source references must correspond to supplied evidence; a generated citation cannot claim an arbitrary reference row.
4. **Provider data exposure:** Generation, translation, semantic grading, and Sense Linking can transmit their task-specific text to external providers. The consuming organization must govern data classification, retention, consent, and provider credentials; this design does not provide a personal-data vault.
5. **Input bounds:** Lexical queries, source handles, citation lemmas, tagged examples, language codes, question sizes, and provider requests have explicit validity/boundedness constraints. Invalid inputs or unusable outputs fail rather than becoming partially trusted content.
6. **Isolation:** A requested Theme is an exact namespace, not a cosmetic toggle over mixed neutral/themed rows. This is a content-isolation rule, not multi-tenant data isolation.

## 7. Integration and operational design

| Area | Design expectation |
| --- | --- |
| Delivery model | Asynchronous Python library invoked by a trusted consumer; not an HTTP API or daemon |
| Required resources | Writable generated-dictionary database and readable Cambridge snapshot; example count passed per generate call, decision threshold and provider key/base URL/model passed explicitly |
| Database targets | PostgreSQL with `pg_trgm` for production generated dictionaries, including an optional dedicated schema; async SQLite for lexical-only local/disposable operation |
| Schema lifecycle | One squashed fresh-schema baseline (`20260930_base`); existing databases need explicit backup, conversion and parity verification before restamping. No legacy upgrades or automatic schema mutation. Migrations never target Cambridge |
| Structured provider | Lazy initialization for tasks that need generation/translation; no provider call is needed for ordinary reads, cached translations, or exact answer matching |
| Decision provider | Lazy initialization for free-text evaluation and eligible Sense Linking; an optional model-backed route is configuration-dependent |
| Resource cleanup | The consumer closes the library when finished; the library closes clients it created, not externally supplied clients |
| Retry / concurrency ownership | Consumer controls overlapping requests, backoff, job scheduling, and throughput; this design provides no global queue or exactly-once job execution |
| Observability boundary | Operations expose results or failures to the consumer; metrics, audit logging, distributed tracing, and alerting are integration responsibilities, not defined service features |

The library and migration runner do not read environment variables or `.env` files.
Consumers pass DB/source/schema, `LLMConfig`, and `DecisionConfig` explicitly.
`generate(..., example_count=5)` controls new examples per Sense, not content already
stored; every namespace retains exactly one Definition. Decision fallback uses the
LLM credentials/base URL with its own configured model ID. Only example scripts
load `.env`, restricted to `LLM_*` and `DECISION_*` provider settings.

## 8. Acceptance and verification matrix

These scenarios are product-level design checks. They are phrased as observable outcomes so a BA can validate intent and a Tester can derive cases without knowing internal modules.

| ID | Scenario | Expected observable outcome |
| --- | --- | --- |
| S1 | Search an existing lemma, alias, inflected form, licensed expression, or approximate spelling | Generated hits are ranked by match class; each reports its matched surface; no pending placeholder is presented as a completed Word |
| S2 | Request available entries for a source label shared by multiple entries | Each eligible unconsumed source entry has a distinct selectable handle; completed associations disappear from the available group |
| S3 | Search a one/two-character query, duplicate surfaces, wildcard-like literal characters, or a late-corpus Word | Short queries have exact/prefix results only; duplicate matches do not consume multiple result slots; literal prefix characters remain literal; corpus position does not limit eligibility |
| S4 | Search an approximate spelling against a SQLite generated store | No fuzzy suggestions or application-level full-dictionary fallback; exact/prefix/pattern behavior remains available |
| G1 | Generate one eligible selected entry twice | One completed Word is returned and reused; no additional Sense inventory is silently created |
| G2 | Generation lacks source evidence, changes the selected lexical item, cites unknown evidence, or returns incomplete content | Request fails; no partially published Word is exposed |
| G3 | Two selected source entries would claim one completed citation identity | Existing completed Word is not overwritten or silently reassigned |
| R1 | Read a Word in neutral and in a Theme without complete themed content | Neutral read succeeds; themed read is unavailable rather than showing neutral material |
| ST1 | Generate Theme content, update Theme metadata, then delete the Theme | Existing themed content remains unchanged on metadata update; deletion removes themed content/questions but preserves neutral Word and questions |
| Q1 | Generate N questions of any supported type with K distractors | All N are saved with K distinct wrong options, or none are saved if the batch is invalid |
| Q2 | Request a question in a Theme namespace, then read/list in neutral | Themed artifacts do not appear in neutral listings; absent themed Word content cannot silently create neutral questions |
| Q3 | Present a question to a learner | Learner sees only safe prompt/options; server retains correct option, explanations, and alternatives |
| GR1 | Submit a valid saved option ID, invalid ID, exact free-text answer, typo, wrong meaning, or valid alternative meaning | Grade and optional Sense ID follow the grading contract; unsupported format and unknown ID are rejected |
| W1 | Resolve a pending relation before and after target Word generation | Pending target is not assumed to be a Sense; only eligible generated target meanings can be considered for Sense Linking |
| W2 | Target definition changes after a relation was resolved | The old resolution is treated as pending until reassessed; independent link failures do not cancel other link outcomes |
| TR1 | Translate identical text twice, then change only whitespace or target language | Identical text/language reuses the cached string; changed text or language creates a distinct translation request |
| DB1 | Delete a Theme; attempt to mutate the source snapshot | Dependency cleanup stays in the generated store; source data remains unchanged |

## 9. Design decisions and known boundaries

- **Selected-source generation over open-ended generation:** Source selection anchors lexical identity and provenance; WordNet can support but does not select or replace the entry.
- **One Word, several Senses:** Meaning-specific content and relations attach to Senses, while aliases and broader lexical links attach to the Word.
- **Neutral-first, exact Theme namespaces:** Theme is an alternate presentation of existing meanings, not an alternate dictionary or a rewrite of neutral material.
- **Saved question artifacts:** A consumer can reuse a question across its supported answer formats and grade against stable saved options. This requires strict separation between trusted storage and learner-facing views.
- **Explicit Sense Linking:** Relation generation creates possible lexical links; Sense matching is a separately requested, potentially costly decision. A resolved Sense link remains dependent on current target meaning content.
- **Exact-text translation cache:** The contract is deliberately narrower than multilingual dictionary localization; callers choose precisely what text to translate.
- **Consumer-owned orchestration:** Selection, rate limits, overlapping writes, retries, privacy policy, learner records, and presentation stay outside the library. Integrations must not assume these concerns are solved internally.

---

**Interpretation note:** This document describes the active system's intended externally visible design and domain rules. It is not a guarantee of production readiness, a bug register, an implementation walkthrough, or a proposal for unimplemented product features.
