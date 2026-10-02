# Examples

Standalone scripts demonstrating the public `Lexicon` API. Run them from the
repository root after installing development dependencies and initializing a
separate generated-content database; see [database setup](../README.md#database-setup).

## Setup

```bash
uv sync --locked
cp .env.example .env
```

Fill in `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`. The `DECISION_*` group is
optional; without a decision key, grading and Sense Linking use the LLM.
Process environment variables override the root `.env` file. Use `--env-file`
to select a different file. Database settings belong in CLI arguments, not `.env`.

For OpenRouter, use `https://openrouter.ai/api/v1` for `LLM_BASE_URL` and
`https://openrouter.ai/api` for `DECISION_BASE_URL`.

The commands below use Bash and an initialized PostgreSQL dictionary:

```bash
DB_ARGS=(--db-url 'postgresql+asyncpg://user:password@localhost/lexicon' \
  --db-schema lexi --cambridge-path data/cambridge.db)
```

Generation, uncached translation, and free-text grading can incur provider charges.
Reads, cache hits, and saved-answer shortcuts do not call providers.

## Dictionary

Search first, then select a reference entry handle from the printed `available` results:

```bash
uv run python examples/01_dictionary.py "${DB_ARGS[@]}" bank
AVAILABLE_ID='selected-available-id'
uv run python examples/01_dictionary.py "${DB_ARGS[@]}" bank \
  --available-id "$AVAILABLE_ID" --target bank --example-count 3
```

Save a Word ID and a Sense ID from the output for the following examples:

```bash
WORD_ID='generated-word-id'
SENSE_ID='selected-sense-id'
uv run python examples/01_dictionary.py "${DB_ARGS[@]}" bank --word-id "$WORD_ID"
```

## Themes

```bash
uv run python examples/02_themes.py "${DB_ARGS[@]}" "$AVAILABLE_ID" \
  --target bank --theme pirate --name Pirate --concept 'A nautical speaking voice'
```

The script creates the Theme if absent and generates or reuses themed content.
Use `--update-name` to change its display name without rewriting saved content.

## Questions

```bash
uv run python examples/03_questions.py "${DB_ARGS[@]}" "$SENSE_ID" \
  definition_to_word --generate-count 2
uv run python examples/03_questions.py "${DB_ARGS[@]}" "$SENSE_ID" definition_to_word
```

Use any [supported question type](../README.md#questions-and-grading).
`--generate-count` appends new artifacts on every call; omit it to read the bank.
Dialogue questions accept `--target-placement dialogue` or `options` during generation.
Use `--theme pirate` only after generating that Sense's themed content.

## Grading

Choose saved Question IDs of the corresponding types. Adapt these answers to the
selected meaning; the sentences below assume the financial noun `bank`.

```bash
uv run python examples/04_grade_single_word.py "${DB_ARGS[@]}" "$SINGLE_WORD_QUESTION_ID" \
  --answer bank --answer bnka --answer chair
uv run python examples/05_grade_definition.py "${DB_ARGS[@]}" "$DEFINITION_QUESTION_ID" \
  --answer 'An institution that holds money and offers financial services.'
uv run python examples/06_grade_usage.py "${DB_ARGS[@]}" "$USAGE_QUESTION_ID" \
  --answer 'I deposited my savings at the bank.' --answer 'We went home.'
```

Example 04 also demonstrates saved-option and exact-word shortcuts. Outputs are
independent diagnostics, not a single learner score. Full Question objects include
answers and explanations and must remain server-side.

## Sense Linking

```bash
uv run python examples/07_sense_linking.py "${DB_ARGS[@]}" "$WORD_ID"
uv run python examples/07_sense_linking.py "${DB_ARGS[@]}" "$WORD_ID" \
  --target-entry "$TARGET_AVAILABLE_ID" "$TARGET" --resolve --batch-size 20
```

Select `TARGET_AVAILABLE_ID` and its lexical `TARGET` from the inspection output.
`--resolve` processes a **global eligible page**, not only the supplied Word's edges.
Use a dedicated example database if other pending work must remain untouched.

## Translation

```bash
uv run python examples/08_translation.py "${DB_ARGS[@]}" \
  --text 'I went to the bank.' \
  --tagged 'I went to the <t inf="base">bank</t>.' --language vi
```

The script repeats the translation to demonstrate cache reuse. Tagged and plain
text share a cache entry when their unwrapped text is identical; whitespace and
target-language changes produce separate entries.

Run any script with `--help` for its full argument list.
