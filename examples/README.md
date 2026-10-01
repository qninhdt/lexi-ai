# Public API examples

Eight independent scripts using `Lexicon`. No shared runner, fake providers,
automatic migrations, database resets or deletion routines. Run from the repository
root with Python 3.14+ and installed project dependencies.

These examples have been authored but **not executed against a database or provider**.

## Configuration

Use a separate generated dictionary database whose schema is already initialized.
The Cambridge snapshot is read-only and must never be the generated database.
See the root [README](../README.md#install) for setup and existing-database cautions.
SQLite supports lexical development reads; PostgreSQL is required for fuzzy search.

Copy `examples/.env.example` to `examples/.env` and fill in provider credentials.
Only two groups of variables are allowed in that file:

```dotenv
LLM_API_KEY=your-llm-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o

DECISION_API_KEY=your-decision-key
DECISION_BASE_URL=https://api.typesafe.ai
DECISION_MODEL=jev-latest
DECISION_FALLBACK_MODEL=gpt-4o
```

`DECISION_FALLBACK_MODEL` is optional; leave it blank to disable fallback. Fallback
uses the **LLM key/base URL**, not the Decision credentials. The tiny `_config.py`
helper belongs only to examples and passes these settings explicitly to `Lexicon`.
It does not mutate the process environment or expand `${VARIABLE}` strings. Explicit
`LLM_*` / `DECISION_*` process variables override the example file. Legacy provider
env names and root/parent `.env` files are not used. `--env-file` selects another
explicit file; unsupported variables in that file are rejected.

DB/source/schema, threshold, content counts and IDs are parameters, **not `.env`
settings**. All scripts require `--db-url`; `--cambridge-path` defaults to the project's
independent `data/cambridge.db` copy. `--db-schema` is optional, `--threshold` defaults to
0.8. Generation scripts accept `--example-count` (default 3 per Sense). Use a Bash
array for repeated CLI arguments without creating any database environment settings:

```bash
DB_ARGS=(--db-url 'postgresql+asyncpg://user:password@localhost/lexicon' \
  --cambridge-path data/cambridge.db --threshold 0.8)
```

The generated database must already be initialized; this array does not create it.
Provider clients are lazy; stored reads, saved option grading
and exact saved single-word answers make no provider requests. Generation, Theme
creation, uncached translation, other free-text grading and eligible Sense Linking
can send data to providers and incur charges. Model judgments are observations, not
assertions about guaranteed semantic outcomes.

## 01 — Dictionary

```bash
uv run python examples/01_dictionary.py "${DB_ARGS[@]}" bank
```

Inspect `words` and `available` separately. Select an `available_id` explicitly;
do not treat the first match as automatically correct. Set the following shell
variables to values you actually selected/read, not the literal placeholders:

```bash
AVAILABLE_ID='selected-available-id'
uv run python examples/01_dictionary.py "${DB_ARGS[@]}" bank \
  --available-id "$AVAILABLE_ID" --example-count 3
WORD_ID='generated-word-id'
SENSE_ID='chosen-sense-id'
uv run python examples/01_dictionary.py "${DB_ARGS[@]}" bank --word-id "$WORD_ID"
```

Generation saves/reuses neutral content; subsequent Word/Sense reads do not generate.
Output includes singular definitions, examples, forms, patterns and relations.

## 02 — Themes

```bash
uv run python examples/02_themes.py "${DB_ARGS[@]}" "$AVAILABLE_ID" \
  --theme pirate --name Pirate --concept 'A nautical speaking voice'
uv run python examples/02_themes.py "${DB_ARGS[@]}" "$AVAILABLE_ID" --theme pirate
# Optional metadata mutation, not a rewrite of stored content:
uv run python examples/02_themes.py "${DB_ARGS[@]}" "$AVAILABLE_ID" \
  --theme pirate --update-name 'Pirate voice'
```

Creates the Theme only if absent, then generates/reuses its exact Word namespace.
Neutral and themed content are displayed separately. Missing themed content does
not fall back to neutral. Existing Theme metadata is reused unless explicitly updated.
`--example-count 5` requests five examples per newly generated Sense. If neutral and
themed content are both new in one call, both get that count; generate neutral first
in example 01 to use a different count for it. Changed counts never rewrite saved content.

## 03 — Questions

```bash
uv run python examples/03_questions.py "${DB_ARGS[@]}" "$SENSE_ID" definition_to_word --generate-count 2
uv run python examples/03_questions.py "${DB_ARGS[@]}" "$SENSE_ID" definition_to_word
uv run python examples/03_questions.py "${DB_ARGS[@]}" "$SENSE_ID" word_to_definition --generate-count 1
uv run python examples/03_questions.py "${DB_ARGS[@]}" "$SENSE_ID" word_to_usage --generate-count 1
uv run python examples/03_questions.py "${DB_ARGS[@]}" "$SENSE_ID" dialogue_completion \
  --generate-count 1 --target-placement dialogue
uv run python examples/03_questions.py "${DB_ARGS[@]}" "$SENSE_ID" dialogue_completion \
  --generate-count 1 --target-placement options
```

All seven types are supported: `definition_to_word`, `context_to_word`, `cloze_to_word`,
`word_to_definition`, `word_to_usage`, `dialogue_completion`, `meaning_in_context`.
Use `--theme pirate` only after generating that Word's themed namespace.
`--generate-count` appends new artifacts **on every invocation**, not a cache lookup.
Without it, the script only lists/gets/retrieves saved Questions. `--limit` and
`--after-id` page the list; random retrieval samples the whole selected bank, not
only that page. Note the printed Question IDs for examples 04–06.

The generation runtime uses two structured shapes: dictionary-anchored answers for
Definition to Word, Word to Definition and Context to Word; model-authored answers
for the remaining types. The public saved Question shape is the same for both.

**Full Question artifacts contain answers and explanations and are server-only.**
The script displays shuffled choice IDs/content without correctness or explanations.
For single-word/short-answer learner views, display only the prompt and Question ID,
not the multiple-choice options. Do not forward full Question objects to learners.

## 04 — Single-word grading

Choose a saved Definition/Context/Cloze to Word Question:

```bash
SINGLE_WORD_QUESTION_ID='chosen-question-id'
uv run python examples/04_grade_single_word.py "${DB_ARGS[@]}" "$SINGLE_WORD_QUESTION_ID" \
  --answer bank --answer bnka --answer chair
```

Those sample words make sense only for a Question targeting `bank`; adapt them to
your prompt. The trusted developer demo submits a saved correct option ID, a saved
distractor ID and the exact saved word, all provider-free. Extra `--answer` samples
use free-text grading unless they match the normalized saved answer.
`task_fit`, `spelling_error` and dictionary `sense_id` are independent outputs.
No combined correctness enum or learner mastery score is constructed.

## 05 — Definition grading

Choose a saved `word_to_definition` Question. For a financial `bank` prompt:

```bash
DEFINITION_QUESTION_ID='chosen-question-id'
uv run python examples/05_grade_definition.py "${DB_ARGS[@]}" "$DEFINITION_QUESTION_ID" \
  --answer 'An institution that holds money and offers financial services.' \
  --answer 'A place for money.' \
  --answer 'An institution that stores money but never lends it.' \
  --answer 'Something somewhere.'
```

Adapt the samples to the actual Word. The Decision first identifies the intended
Sense from the owner Word's entire neutral inventory, then judges `accuracy` and
`coverage`. The identified Sense need not be the original Question Sense. No
identifiable Sense yields null diagnostics; it is not an accuracy verdict.

## 06 — Usage grading

Choose a saved `word_to_usage` Question. For the financial noun `bank`:

```bash
USAGE_QUESTION_ID='chosen-question-id'
uv run python examples/06_grade_usage.py "${DB_ARGS[@]}" "$USAGE_QUESTION_ID" \
  --answer 'I deposited my savings at the bank.' \
  --answer 'I deposited my savings at the bnka.' \
  --answer 'I bank my savings at yesterday.' \
  --answer 'We sat on the river bank.' \
  --answer 'We went home.'
```

Observe `used`, then `meaning`, `form`, `construction`, `collocation`, `appropriacy`.
When `used=false`, all five diagnostics are null. Evaluation uses the Word/meaning
anchor saved in the Question, not newly edited dictionary text. Samples intentionally
include different meanings and errors; no exact model verdict is assumed.

## 07 — Sense Linking

```bash
uv run python examples/07_sense_linking.py "${DB_ARGS[@]}" "$WORD_ID"
# Select target handles from the printed Cambridge hits yourself:
TARGET_AVAILABLE_ID='selected-target-available-id'
uv run python examples/07_sense_linking.py "${DB_ARGS[@]}" "$WORD_ID" \
  --target-entry "$TARGET_AVAILABLE_ID" --resolve --batch-size 20
```

Default behavior is inspection/search only. Repeated `--target-entry` explicitly
generates selected targets. Generation alone never resolves relations.

**`--resolve` processes a global eligible page, NOT just the supplied Word's edges.**
`word_id` only selects which source relations to inspect before/after. Use an example
database when you do not want other pending work processed. There is no hidden loop
draining the queue and no invented source filter.

Each candidate must satisfy target-gloss compatibility and directional relation
validity. The operation includes every same-POS candidate, not a capped inventory.
Per-edge results distinguish `resolved`, `unresolvable`, `error`, `noop`; stored
`resolution_state` is `pending`, `resolved` or `unresolvable`. Failed edges stay
pending; unavailable targets are deferred, not negative decisions. Concurrent edits
can prevent applying stale evidence without erasing independent successful links.

## 08 — Translation

```bash
uv run python examples/08_translation.py "${DB_ARGS[@]}" \
  --text 'I went to the bank.' \
  --tagged 'I went to the <t inf="base">bank</t>.' --language vi
# Optional additional cache misses:
uv run python examples/08_translation.py "${DB_ARGS[@]}" \
  --text 'I went to the bank.' --whitespace-variant --other-language fr
```

The script translates twice to demonstrate reuse, optionally translates valid tagged
text, and lists a bounded page of saved translations. Tagged/plain inputs share a
cache entry only when their exact unwrapped text and language match. Whitespace
remains significant; different target languages have separate keys. Equal translated
wording is not proof of cache identity. `--translation-id` reads a saved record;
`--after-id` pages the cache list. No deletion/purge is performed.

Every script uses the variable `lexicon` and closes it in `finally`. Exceptions are left visible rather
than disguised as successful negative semantic results. Retries, concurrency and
learner progress belong to the consuming application.
