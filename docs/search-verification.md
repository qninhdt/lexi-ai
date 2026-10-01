# PostgreSQL search verification — 30 September 2026

## Outcome and scope

The active dictionary uses PostgreSQL-native trigram search on normalized lemma,
alias, and Sense-form keys. Correctness tests verify the agreed match-class order,
best-match-per-Word deduplication before the result limit, literal prefix escaping,
Unicode normalization, unpublished-Word exclusion, short queries, and Sense-scoped
expression patterns. SQL tests ensure there is no Python fuzzy scan, KNN operator,
or arbitrary raw-candidate LIMIT. An isolated installed wheel also verifies the
`lexi_ai` / `Lexicon` surface and packaged Jinja prompts.

The measured corpus contains **100,000 Words**, **4,776 aliases**, **8,422 forms**,
and one minimal Sense per Word. Words are a deterministic sample spread across
the full read-only Cambridge snapshot after citation validation and normalized
identity deduplication. Aliases and forms come from that snapshot; no LLM is called.
This is a retrieval fixture, **not an import of generated dictionary content**.

## Environment and measurements

- PostgreSQL **18.6**, disposable Docker database on a local loopback connection.
- Python **3.10.21** in the recorded benchmark; asynchronous public search path, including transactions,
  SQL execution, result assembly, and the separate empty pattern-table query.
- `shared_buffers=128MB`, `work_mem=4MB`, `enable_seqscan=on`.
- `gin_fuzzy_search_limit=0`; no index-forcing planner settings.
- Twenty serial warm-cache requests per case, after a warm-up request.
- Seed/COPY plus `VACUUM (ANALYZE)`: **3.30 seconds** (excludes reading/normalizing
  the source snapshot and creating the empty schema).

| Case | Query | Results | Median ms | p95 ms |
| --- | --- | ---: | ---: | ---: |
| Exact lemma | `bank` | 30 | 7.656 | 9.093 |
| Lemma/prefix | `trans` | 30 | 8.406 | 10.504 |
| Exact alias among lemma/fuzzy hits | `walking papers` | 30 | 23.643 | 25.573 |
| Exact form among lemma/prefix/fuzzy hits | `children` | 30 | 14.260 | 35.587 |
| Fuzzy lemma | `dictionry` | 11 | 24.115 | 28.479 |
| Fuzzy alias | `walking paperss` | 30 | 32.115 | 46.633 |
| Fuzzy form | `childrenn` | 30 | 26.530 | 33.633 |
| Broad one-character prefix | `a` | 30 | 25.027 | 29.269 |
| Two-character exact/prefix | `ta` | 30 | 11.274 | 19.800 |
| Common typo | `bankk` | 30 | 25.832 | 30.021 |
| No match | `zzqxjjvv` | 0 | 10.852 | 13.790 |

The label describes a tested source, not an exclusive result class. For example,
`children` returns the lemma `children` before the form `children` of `child`.
Exact queries may also perform fuzzy filling when fewer than 30 lexical/pattern
Words exist; their latency is not a measurement of equality SQL alone.

The three GIN index sizes were **7,364,608 bytes** (Words), **1,490,944 bytes**
(aliases), and **1,712,128 bytes** (forms).

## Actual planner evidence

`EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)` was collected for the SQL issued by the
public search path, including bound parameters. Exact/prefix plans used the
normalized-key B-tree indexes. Word fuzzy branches used the partial GIN index
with the required `generation_state='done'` predicate. Form branches used their
GIN index, then joined through Sense ownership. Alias plans sometimes used GIN
and sometimes scanned the small 4,776-row alias table; the planner was not forced.
There was no full Word-table scan in the measured **runtime fuzzy plans**.

A final plan-only smoke run confirmed each GIN source with natural planner choices:

| Source probe | Index | Index candidates | Rows after recheck | Shared buffer hits | SQL execution ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Word `dictionry`, published only | `ix_words_match_key_trigram` | 1,432 | 11 | 487 | 4.011 |
| Alias `walkingg` | `ix_word_aliases_match_key_trigram` | 6 | 2 | 24 | 0.059 |
| Form `childrenn` | `ix_sense_forms_match_key_trigram` | 31 | 11 | 29 | 0.107 |

For `walking paperss`, PostgreSQL instead selected an alias sequential scan,
returning 3 matches after rejecting 4,773 rows (7.544 ms in that smoke probe).
This is a database cost-based plan, not an application fallback. GIN is not a
guarantee that every query uses an index or that only 30 rows are processed.
For `bankk`, the runtime Word GIN branch returned 2,208 raw candidates and retained
72 after recheck; the union across sources yielded 80 rows before per-Word ranking
and final limiting. No candidate cap was added to hide this work.

## Correctness and known limits

- `walkng` gives `walk` and `walking` the same native similarity in the tested
  PostgreSQL version; deterministic surface tie-breaking can select `walk`.
- `dictionry` ranks `diction` ahead of `dictionary`. Trigram similarity is not
  edit distance, semantic equivalence, or a promise of the user's intended word.
- A 250-row duplicate-form fixture verifies that repeated matches for one Word
  cannot consume the final result slots. Existing lexical/pattern Words are
  excluded from fuzzy filling so they cannot crowd out other suggestions.
- Tests verify transaction-local similarity threshold **0.3** and uncapped GIN
  retrieval even when a pooled connection previously had different settings.
- One/two-character queries have exact/prefix retrieval only. Pattern matching
  remains separate, anchored, and limited to licensed forms of the same Sense.
- SQLite has no fuzzy implementation and no full-dictionary fuzzy fallback.
- Jinja templates are split at exactly one system marker followed by one user
  marker **before interpolation**. User/source text cannot create another role
  by containing marker-like strings. Role separation is not a complete prompt-
  injection defense and does not replace provider-output validation.

## Reproduction and evidence

Use a **disposable** PostgreSQL URL with schema-creation and `pg_trgm` privileges:

```bash
uv run pytest -q tests
uv run python tests/benchmark_postgres_search.py \
  --source /path/to/cambridge.db \
  --output /tmp/opencode/lexi-search-benchmark.json
```

Both commands use `LEXI_TEST_PG_URL`; `LEXI_REQUIRE_PG=1` prevents silently skipping
PostgreSQL checks in CI. The benchmark creates, populates, and removes only its
own randomly named schema. The reference snapshot is opened read-only.

The local 20-request evidence is `/tmp/opencode/lexi-search-benchmark.json`;
the final one-request plan smoke is `/tmp/opencode/lexi-search-benchmark-final.json`.
These are local artifacts, not repository resources; rerun the script to regenerate
complete SQL, parameters, plans, buffers, and timing results elsewhere.

**Not measured:** concurrent load, cold-cache/disk behavior, sustained production
traffic, generated content quality, large pattern inventories, or production-like
Sense/alias/form fan-out. The large probe has **zero patterns** and one synthetic
Sense per Word; pattern correctness is covered by dedicated tests, not this scale
measurement. These figures are environment-specific evidence, not a latency SLO
or a claim that all production workloads are optimized.
