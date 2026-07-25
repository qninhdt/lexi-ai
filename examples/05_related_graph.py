"""Trial 05 — lazy graph: related words become stubs, generated on demand.

When an entry is generated, its ``related`` words are persisted as ``pending``
stubs (no senses yet, zero extra tokens). Querying one of those stubs later
triggers its own one-time generation, turning it ``done``. This walks one hop of
that graph live.

    uv run python examples/05_related_graph.py [word]
"""

import asyncio
import sys

from _common import aclose, build_lexicon, lookup, print_entry
from sqlalchemy import select

from lexi_ai.db import session_scope
from lexi_ai.infrastructure.db.models import Word


async def _status_of(lex, norm: str) -> str | None:
    from lexi_ai.normalize import match_key

    async with session_scope(lex._session_factory) as session:
        row = await session.execute(select(Word.status).where(Word.match_key == match_key(norm)))
        return row.scalar_one_or_none()


async def main(word: str) -> None:
    lex = build_lexicon()
    await lex.init()
    try:
        print(f"=== Generate {word!r} ===")
        entry = await lookup(lex, word)
        if entry is None:
            print("(no Cambridge match — try another word)")
            return
        print_entry(entry)

        if not entry.links:
            print("\n(no related words emitted for this entry — try another word)")
            return

        # Pick the first related word and show it is a pending stub.
        target = entry.links[0]
        print(f"\nrelated words -> {[ln.display for ln in entry.links]}")
        status = await _status_of(lex, target.norm)
        print(f"stub {target.display!r} current status: {status!r}  (should be 'pending')")

        print(f"\n=== Query the stub {target.display!r} (pending -> generate on demand) ===")
        sub = await lookup(lex, target.norm)
        if sub is not None:
            print_entry(sub)
            status_after = await _status_of(lex, target.norm)
            print(f"\nstub status after lookup: {status_after!r}  (now 'done')")
    finally:
        await aclose(lex)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "happy"))
