# CLI recipes

Install the package with `uv sync --locked` in a checkout or `pip install .`.
Use `lexi` (or `python -m lexi_ai`); all commands ship in the installed package.
Copy the root [.env.example](../.env.example) to `.env` in the current directory for
provider operations. Process environment overrides it; `--env-file PATH` selects
another file. Reads/imports need no provider credentials. Generation, uncached
translation and model-based grading can incur provider charges.

## Generate and import

```bash
lexi generate word bank --output content.sqlite
lexi generate word --word-list words.txt --output content.sqlite
lexi generate question 1 --output content.sqlite --type DEFINITION_TO_WORD
lexi generate word --word-list words.txt --output 'postgresql://user:password@localhost/lexicon'
lexi import content.sqlite --into 'postgresql://user:password@localhost/lexicon'
```

TXT accepts UTF-8/BOM, one word per line; blanks and identical duplicates are ignored.
Word generation runs up to `--max-concurrency 128` words concurrently and returns
complete Words. Questions take a saved Sense ID, one required `--type`, and `--count`
new Questions. Fixed-stem types (`DEFINITION_TO_WORD`, `WORD_TO_DEFINITION`,
`WORD_TO_USAGE`) default to 1 Question and 5 distractors; the other types default
to 8 Questions and 3 distractors. Explicit count flags override these defaults.
They return full Questions and
never generate Words. `--theme KEY` selects existing themed content;
`--target-placement` applies to dialogue questions.

Generation initializes the current database. Word generation downloads/caches reference
as needed; `--reference-path reference.sqlite` selects an offline source. Existing Words
are reused. Failed Words continue with a nonzero exit and retain successful work.
SQLite output is finalized with a checksum sidecar. Import preserves IDs and
refuses conflicting rows; `--words-only` excludes saved questions.

The checkout keeps its reference snapshot at `data/reference.sqlite`. To build
words and all seven question types from a common-word list into the same directory:

```bash
uv run python scripts/generate-content.py --word-list /path/common-words.txt
```

Run from the checkout root with provider settings in `.env`. The script uses one
Lexicon and runs up to 128 Words concurrently. Each Word immediately starts all
seven Question types per Sense concurrently; tqdm advances per completed word.
Defaults are 128 concurrent provider requests, 1 Question with 5 distractors per
Sense/fixed-stem type, and 8 Questions with 3 distractors per Sense/context type.
`--count` applies to the four context types. `--output`,
`--reference-path`, `--env-file`, `--count`, and `--max-concurrency`
override defaults. Failures are printed and return exit code 1; other words continue.
Use `lexi generate word` to generate only dictionary content and `lexi generate
question` for Questions of existing words. Ctrl+C preserves each committed Word
and question bank; unfinished operations roll back. Run the same command again to
reuse saved words and fill missing questions. After a hard stop, keep any
`content.sqlite-wal` file with the database until the next run recovers/finalizes it.
The script checkpoints SQLite and writes `data/content.sqlite.sha256` on orderly
exit, including cancellation. A hard stop may leave WAL pending until the next
run. Generated content is Git-ignored.

```bash
lexi db init --db content.sqlite --reference-path reference.sqlite
lexi db init --db 'postgresql://user:password@localhost/lexicon'
lexi db head
lexi db current --db content.sqlite
lexi import reference.sqlite --kind reference --into 'postgresql://user:password@localhost/lexicon'
```

PostgreSQL data survives process/container restarts in persistent storage. Import is
setup work; ordinary reads never import/migrate. SQLite artifacts are not runtime
mounts for Pycil. `--db-schema` selects a PostgreSQL dictionary schema only.
SQLite reference search/validation opens the cached reference automatically;
`--reference-path reference.sqlite` selects an offline file for those commands.

## Saved dictionary and exercises

Replace sample numeric IDs with IDs returned by your generation commands.

```bash
lexi word search bank --db content.sqlite
lexi word search bank --include-reference --db content.sqlite
lexi word get 1 --db content.sqlite
lexi sense get 1 2 --db content.sqlite
lexi sense previews 1 2 --db content.sqlite
lexi reference validate SELECTED_REFERENCE_ID --db 'postgresql://user:password@localhost/lexicon'
lexi question get 1 2 --db content.sqlite
lexi question list 1 --type DEFINITION_TO_WORD --limit 20 --db content.sqlite
lexi question list 1 2 --limit-per-type 8 --db content.sqlite
lexi question count 1 2 --db content.sqlite
lexi question retrieve 1 --type DEFINITION_TO_WORD --db content.sqlite
lexi question retrieve --request 1:DEFINITION_TO_WORD:2 --request 2:WORD_TO_USAGE:1 --db content.sqlite
lexi question grade 1 --format SINGLE_WORD --answer bank --db content.sqlite
lexi question delete 1 --db content.sqlite
```

Question artifacts include answers/explanations; consumers decide answer visibility.
Grading returns separate diagnostics, not a single learning score. Saved option IDs
and normalized saved-word answers can grade locally. `--mode` uses native uppercase
`LLM_FALLBACK`, `LLM_ONLY` or `DECISION_ONLY`; `--allowed-pair TYPE:FORMAT` limits
accepted grading formats. `--with-usage` on inference commands returns usage records.
Long answers or translation input can use `--answer -` / `--text -` for stdin.

## Themes, relations and translations

```bash
lexi generate theme pirate --name Pirate --concept 'A nautical voice' --output content.sqlite
lexi theme get pirate --db content.sqlite
lexi theme list --limit 20 --db content.sqlite
lexi theme update pirate --name 'Pirate voice' --db content.sqlite
lexi generate word bank --theme pirate --output content.sqlite
lexi generate question 1 --type DEFINITION_TO_WORD --theme pirate --output content.sqlite --count 8
lexi relation resolve --batch-size 20 --db content.sqlite
lexi generate translation --text 'I went to the bank.' --language vi --output content.sqlite
lexi translation list --limit 20 --db content.sqlite
lexi translation get 1 --db content.sqlite
lexi translation delete 1 --db content.sqlite
lexi translation purge --db content.sqlite
lexi theme delete pirate --db content.sqlite
```

Theme namespaces never substitute neutral content. Relation resolution processes
**global eligible links** in the database, not only a selected word. Delete/purge
commands perform their explicit destructive operation. Lists support `--after-id`
(or theme `--after-key`). Progress/errors go to stderr; JSON results go to stdout.
Exit codes: 0 success, 1 operation/batch failure, 2 argument error, 130 interruption.
