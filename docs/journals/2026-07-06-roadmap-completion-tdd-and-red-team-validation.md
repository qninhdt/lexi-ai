# 10-Phase Roadmap Complete: TDD-First Prevents Cache Poisoning, Scouting Shrinks Scope

**Date**: 2026-07-06 14:00
**Severity**: Low (successful delivery)
**Component**: Cache layer, question engine, LLM client, TTS integration, semantic relations
**Status**: Resolved

## What Happened

Completed a 10-phase implementation roadmap for lazy-generation content (translations, TTS, semantic relations, question formats). All landed on main (commit 3268c98), 307 tests passing (up from 256 at roadmap start), ruff clean, no blocking issues.

## The Satisfaction

The discipline of TDD-before-refactor and scouting-before-building paid off. A red-team review caught a production-risk cache design flaw *before* shipping, and we redesigned it under test. Then a scout spike on idioms/phrasal verbs discovered they were already learnable content—the "missing feature" was just a 2-column view ergonomic. This kind of validation saves rework and prevents half-built systems.

## Technical Details

**Cache redesign (TDD-first)**: Reference-addressed asset identity shifted from pure content hash to a source tuple `(source_kind, source_id, kind, params)` with a stored `content_hash` verified at read time. Red-team feedback: keying on mutable `sense_id` could serve stale content if rowids got reused. New model: derived assets live by their source reference, but validation happens on retrieval. Cache misses cleanly when a source changes or is regenerated; cascade cleanup is best-effort (correctness falls out of the read-time hash verify, never serving stale data). Characterized current behavior under test before touching a line.

**Selective anchoring**: Stopped hard-anchoring semantic relations to Cambridge/WordNet (they're now LLM-generated); instead anchored IPA pronunciation from Cambridge on senses, stored per-POS (ipa_uk, ipa_us), sanitized NUL-safe for Postgres. This cut a mutable reference problem and aligned with memory constraints (no reference-reuse gotchas).

**LangChain removal**: Deleted before TTS work so new integrations shipped on the final client (openai SDK only). One dependency path instead of a chain of indirect compatibility concerns.

**TTS with credential safety**: POST to OpenAI-compatible /audio/speech endpoint via openai SDK. When api_key is set, base_url must be https:// or loopback, or construction fails loudly—never ships cleartext credentials. Unconfigured, stub provider raises (no fabricated audio). Questions degrade gracefully to no-TTS variants.

**Question engine plugins**: Three new formats (`matching`, `listening`, `spelling`) as plugin dispatch—one format = one plugin owning generate + grade. Engine never branches on backend. `listening` and `spelling` synthesize audio via TtsPort; payload stores durable AudioRef (source tuple, not rowid), survives purge/regenerate. Grading index/text-based, never touches the clip.

**Idioms discovery**: Scout spike found they were already first-class learnable content (entry_type preserved, full senses generated, accessible from host word). Missing piece: the link view on host entries didn't carry word_id + status for safe lookup. Fix was additive—two eager-loaded scalar columns, no lazy-load surprise. Prevented building a parallel idiom-specific system.

## What We Tried

- **Initial cache design**: Pure content-addressed keyed on `sense_id`. Red-team flagged: sense_id can be reused across regenerations, rowid collisions possible → stale content served. Discarded, redesigned around source reference.
- **Idiom feature**: Spec'd a custom learnable-phrases system before scoping idioms. Scout revealed they were already built in; scope shrank to view fix. Avoided months of work.

## Root Cause Analysis (Why This Matters)

The red-team cache catch happened because we committed to characterization tests *before* refactoring—we locked current behavior first, then changed the identity model, and tests caught edge cases immediately. No TDD = shipping poisoned cache to production. The idiom scout worked because we asked "is this already in the system?" before designing new code—the feature existed, just undiscovered.

The pattern: upstream validation (red-team design review + scouting spike) + downstream coverage (TDD on sensitive changes) = fewer surprise reworks and fewer false features.

## Lessons Learned

1. **Characterization tests are insurance**: Before refactoring cached/mutable state, lock current behavior with tests. When you change the contract (identity model), tests catch poisoning immediately.
2. **Red-team review worth the cycle**: A second pair of eyes asking "what if rowid gets reused?" saved a production bug. Make time for it before shipping mutable state.
3. **Scout before scope**: Ask "is this already built?" before designing new systems. Idioms saved us weeks.
4. **One-dependency-path wins**: Removing LangChain before TTS meant no "wait, which SDK supports this?" compatibility ladder. TTS shipped on the final client from day one.
5. **Plugin dispatch > branching**: Question formats as discrete plugins (not if/else chains) kept the engine simple and made each format testable in isolation.

## Next Steps

- Monitor cache behavior in production; if hash mismatches spike, investigate upstream source mutation or rowid reuse patterns.
- Gather telemetry on TTS uptake; unconfigured stub should raise loudly so we know if users are hitting degraded mode.
- Follow up on remaining edge cases flagged in review (optional, post-ship backlog).
