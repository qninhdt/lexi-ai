# Phases 2-4: concurrency, content integrity, refactor, dual-DB verification

**Date:** 2026-07-11
**Plan:** `260711-audit-driven-refactor` / Phases 2, 3, 4
**Tests:** 443 passed (both tiers: 435 hermetic + 8 Postgres) — 0 skipped when the PG tier runs
**Lint:** `ruff check` + `ruff format --check` clean

---

## What was done

Phases 0-1 were already committed (`65b7165`). This session landed Phases 2-4
plus the finalize sync-back, and ran the full dual-DB tier against a real
Postgres 16 container (`LEXI_TEST_PG_URL` on :5433).

### Phase 2 — concurrency and content integrity

- **2.1 (baseline):** the suggestion arm already double-checks by Cambridge
  provenance (`_generated_by_cambridge`), not `match_key(display)` — the
  re-scoped fix was present. No change needed.
- **2.3 — TTS voice/fmt allow-list:** added `TTS_VOICES`/`TTS_FORMATS` to
  `constants.py`; `normalize_asset_params` now resolves `None`→config default
  BEFORE validating both params against the closed vocab (mirrors `lang`). Closed
  the filename-collision (`en-US` vs `en_US`) + identity-separator ambiguity.
  Updated `test_tts_params_stable` to the resolved-default contract; relocated the
  path-traversal regression to feed `put_file` a raw crafted `params` directly so
  its coverage survives the upstream gate. Asserted the config defaults are vocab
  members.
- **2.4 — numeric theme names:** `resolve_theme(str)` now tries `theme_key` FIRST,
  falling back to `int()`-by-id only on a key miss. A theme named "1984" resolves
  by name; a stringified id "42" still resolves by id. Widened `generate`/
  `generate_many` `theme` param to `str | int | None` (backward-compatible, aligns
  with `get_entry`).
- **2.5 — WSD docstring:** scoped the `_align` guarantee to COUNT only (docstring
  only; no code/schema change).
- **2.6 — themed single-flight:** extended single-flight to the theme overlay,
  keyed on `(word_id, theme_id)` via a distinct `_theme_locks` map, with an
  in-lock overlay re-check so the second concurrent waiter adopts the first's
  overlay instead of a duplicate LLM call.

### Phase 3 — refactor and dead-code cleanup

- **3.1 (security-High):** routed all three themed generation call sites through
  `guarded_messages` (nonce-wrapped) — the user-controlled `style_prompt` was the
  one LM surface bypassing the injection guard.
- **3.2:** made `constants.py` the single source for `WSD_BATCH_CEIL`/
  `WSD_CANDIDATE_CAP`; deleted the dead `wsd.py:WSD_CANDIDATE_CAP` and the
  duplicate literals.
- **3.3/3.4/3.5 (questions):** extracted a shared `DistractorDedup` (exclude+dedup
  used by both the ladder provider and `_merge_distractors`); consolidated the two
  token-blank loops into `_blank_first_token_matching` with two behavior fixes
  (separator preservation, unicode İ alignment) each carrying failing-first tests;
  passed `format` into `_mcq_question` so callers stop mutating a constructed read
  model.
- **3.6:** extracted `_resolve_theme_or_raise` (the three-copy theme resolve).
  `_clean_opt` was already extracted in Phase 1.
- **3.7 (owner: fix it):** `get_senses` now populates `SenseView.relations`
  (eager-loaded `relations_out` + target word/sense), closing the latent read-model
  gap vs `_build_entry`.
- **3.8:** `put_file` now unlinks a just-written orphan on non-IntegrityError
  failure (only when it created the file); other edges documented or confirmed
  non-issues.
- **3.9:** dead-branch items (`_wrap_untrusted` `max_len`, `ainvoke_structured`
  `max_retries=0`) fixed docstring-only, no behavior-change code.
- **3.10 (owner: keep fail-closed):** regenerate-failure demote left as-is,
  tradeoff documented in `_record_error`.

### Phase 4 — verification and the Postgres CI gap

- Dual-DB regressions on the Phase 0 harness for **all five untrusted columns**
  (`norm`, `alias_norm`, `source_ref`, `definition`, `example`) + over-length
  `source_ref` + the H2 tz-aware datetime bind (no re-judge loop).
- Ran the full tier against a real Postgres 16 container: **8/8 pass**. Confirmed
  the teeth are real (Postgres rejects raw NUL with a DBAPIError; the cleaned
  value succeeds).
- Backend-agnostic regressions (H1, TA1, TA2, injection) already present on the
  default tier from Phases 1-3.
- Updated `docs/system-architecture.md` Testing section (test count + dual-DB
  tier invocation).

---

## Code review + finalize

A `code-reviewer` pass ran against the full diff with the plan's acceptance
criteria. Verdict: build/lint/tests green, all HIGH findings fixed with
red-before-green teeth, injection fix and dual-DB class correctly closed. Two
actionable findings resolved:

- **M1:** the `theme` widening contradicted plan.md's acceptance criterion 4;
  reconciled plan.md to record the backward-compatible widening as accepted
  (Phase 2 §2.4 owns the decision).
- **L2:** restored a `_count == 1` assertion accidentally dropped from
  `test_insert_word_recovers_from_concurrent_duplicate` when the new NUL block was
  appended.

Plan + all phase files synced to Done; acceptance criteria checked off.

---

## Key decisions

- Distinct `_theme_locks` map (not the neutral `_locks`) so the overlay lock can
  never deadlock against the per-key generate lock.
- `_drop_format_chars` folds Unicode category `Cf` only — the placeholder
  sentinels are `Co` (private-use), so they survive.
- CI Postgres job not added (no workflow file); the manual `LEXI_TEST_PG_URL`
  invocation is documented, which the Phase 4 criterion allows as the fallback —
  infra gap flagged to the owner.

---

## What's next

- Optional: add a CI Postgres service-container job so the dual-DB tier runs on
  every push (currently opt-in / manual).
