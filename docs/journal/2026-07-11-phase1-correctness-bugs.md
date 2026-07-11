# Phase 1: Correctness bugs — dual-DB fixes, crash guards, refactor

**Date:** 2026-07-11  
**Plan:** `260711-audit-driven-refactor` / Phase 1  
**Commit:** `65b7165`  
**Tests:** 431 passed, 1 skipped (Postgres integration — no `LEXI_TEST_PG_URL`)  
**Lint:** `ruff check` + `ruff format --check` clean  

---

## What was done

Phase 0 (Postgres test harness) was already committed. Phase 1 addressed five
confirmed defects — all invisible to the SQLite-only CI, visible on Postgres.

### 1.1 — `grade_single_choice` crash on adversarial answer

The old index parse used `text.lstrip("-").isdigit()` followed by `int(text)`.
`str.isdigit()` accepts Unicode superscripts (²) and `lstrip("-")` allows "--5"
— both pass the guard but crash `int()`. A 5000-digit string trips CPython's
`int_max_str_digits` limit. Fixed with `_ASCII_INT_RE = re.compile(r"^-?[0-9]+$")`
plus `try/except ValueError` returning `None`.

### 1.2 — tz-aware datetime silent re-judge loop on Postgres

`_mark_unresolvable` stamped `datetime.now(timezone.utc)` (aware) onto a
`TIMESTAMP WITHOUT TIME ZONE` column. asyncpg raises DataError; the swallowed
exception left edges as derived-pending, re-queued forever. Fixed with `_utcnow()`
(naive UTC). Regression captures the bound value at cursor time (not read-back).

### 1.3 — Untrusted LM columns bypassing `_clean`

Neutral path wrote definition and examples raw; norm, alias_norm, source_ref were
never cleaned on any path. A NUL crashes the Postgres INSERT for the whole word.
Fixed by routing all five columns through `_clean()`. Added `max_length=255` to
`GeneratedReference.source_ref` in the Pydantic schema.

### 1.4 — `match_key` missing control/NUL + zero-width stripping

`match_key` did not apply `_CTRL_RE` or strip zero-width/format chars (ZWSP, BOM,
soft-hyphen — Unicode category Cf). Both caused Postgres-only bugs. Fixed with
`_CTRL_RE.sub(" ", s)` + `_drop_format_chars(s)` before placeholder folding.
Documented as pre-population-only.

### 1.5 — `_clean_opt` refactor

Extracted `@classmethod _clean_opt(cls, s, cap)` from six copy-pasted idioms.
No `_base_fold` helper — intentional per-key duplication preserved.

---

## Key decisions

- No hash-suffix-on-truncation for over-length `match_key` (fail-closed convention preserved)
- Naive UTC via `_utcnow()` only
- Inline fixes (1.3/1.4) before refactor (1.5) so revert cannot re-open bugs
- Lint fix in `contextual_mcq.py` (pre-existing E501 in working tree)

---

## What's next

- **Phase 2:** Concurrency and content integrity
- **Phase 3:** Refactor and dead-code cleanup
- **Phase 4:** Verification and Postgres CI gap
