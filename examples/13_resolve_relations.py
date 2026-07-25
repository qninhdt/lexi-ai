"""Trial 13 — WSD sense-relation resolution on a LIVE model (Phase 4).

Sense-level relations (synonym/antonym/hypernym/...) are emitted at generation
time as HALF-edges: ``from_sense -> (to_word, gloss)``, where ``gloss`` describes
the intended MEANING of the target. A later pass reconciles each half-edge to a
concrete target SENSE (word-sense disambiguation) using an LLM judge:

  1. lift derived-``pending`` edges whose target word is ``done`` with senses,
  2. POS-filter the target's candidate senses (both sides normalized),
  3. batch-judge (one prompt, N tasks) which target sense the gloss points to,
  4. write ``to_sense_id`` (+ a content hash) for a pick, or stamp
     ``resolve_attempted_at`` for a "none" (derived ``unresolvable``).

This trial runs the judge against the REAL configured model (unlike the hermetic
pytest suite, which fakes it) so you can eyeball WSD quality end to end. Generate
a couple of related words first so there are inbound edges to resolve; the
inbound-resolve hook already fires during generation, so this manual batch mostly
mops up anything generated before the feature existed.

    uv run python examples/13_resolve_relations.py [source_word] [target_word]

Needs the LLM env (LEXI_LLM_*). Spends tokens on the first generation of each
word plus one judge call per batch.
"""

import asyncio
import sys

from _common import aclose, build_lexicon, lookup
from sqlalchemy import select

from lexi_ai.db import session_scope
from lexi_ai.infrastructure.db.models import Sense, SenseRelation, Word

DEFAULT_SOURCE = "bright"
DEFAULT_TARGET = "dark"


async def _dump_edges(lex) -> None:
    """Print every sense_relation with its derived state (Q1 — no state column)."""
    sf = lex._session_factory  # example-only peek at the raw edges
    async with session_scope(sf) as s:
        rows = (
            await s.execute(
                select(
                    SenseRelation.rel_type,
                    SenseRelation.gloss,
                    SenseRelation.to_sense_id,
                    SenseRelation.resolve_attempted_at,
                    Word.norm,
                )
                .join(Word, Word.id == SenseRelation.to_word_id)
                .order_by(SenseRelation.id)
            )
        ).all()
        if not rows:
            print("  (no sense-level relations emitted)")
            return
        for rel_type, gloss, to_sense_id, attempted, target_norm in rows:
            if to_sense_id is not None:
                state = "resolved"
            elif attempted is not None:
                state = "unresolvable"
            else:
                state = "pending"
            target_def = ""
            if to_sense_id is not None:
                target_def = (
                    await s.execute(select(Sense.definition).where(Sense.id == to_sense_id))
                ).scalar_one_or_none() or ""
            arrow = f" -> {target_def!r}" if target_def else ""
            print(f"  [{state:12}] {rel_type} {target_norm!r} (gloss: {gloss!r}){arrow}")


async def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
    target = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TARGET
    lex = build_lexicon()
    await lex.init()
    try:
        # Generate the source (emits half-edges) then the target (its inbound-resolve
        # hook fires during generation). We still run the manual batch below to show
        # the API and catch anything the hook missed.
        print(f"=== generating source {source!r} (emits sense-relation half-edges) ===")
        if await lookup(lex, source) is None:
            print(f"◇ {source!r} not in Cambridge — try another word.")
            return
        print(f"=== generating target {target!r} (inbound-resolve hook fires here) ===")
        if await lookup(lex, target) is None:
            print(f"◇ {target!r} not in Cambridge — try another word.")
            return

        print("\n--- edges after generation (hook already ran) ---")
        await _dump_edges(lex)

        print("\n=== manual resolve_relations() batch (mop-up / live judge) ===")
        results = await lex.engine().resolve_relations(batch_size=20)
        if not results:
            print("  (no pending edges left — the inbound hook resolved them all)")
        for r in results:
            outcome = r.value if r.ok else f"error: {r.error}"
            print(f"  edge {r.key}: {outcome}")

        print("\n--- edges after manual batch ---")
        await _dump_edges(lex)
    finally:
        await aclose(lex)


if __name__ == "__main__":
    asyncio.run(main())
