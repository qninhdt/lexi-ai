"""Trial 01 — look up a single word, then prove the cache is free.

First call runs the full lazy pipeline (Cambridge + WordNet -> LLM -> DB).
Second call reads straight from the DB: zero tokens, same Entry.

    uv run python examples/01_lookup_word.py [word]
"""

import asyncio
import sys
import time

from _common import aclose, build_lexicon, lookup, print_result


async def main(word: str) -> None:
    lex = build_lexicon()
    await lex.init()
    try:
        print(f"=== 1st lookup: {word!r} (miss -> generate) ===")
        t0 = time.perf_counter()
        first = await lookup(lex, word)
        print(f"[{time.perf_counter() - t0:.2f}s]")
        print_result(first)

        print(f"\n=== 2nd lookup: {word!r} (cache hit -> no LLM) ===")
        t0 = time.perf_counter()
        second = await lookup(lex, word)
        print(f"[{time.perf_counter() - t0:.2f}s]")
        print_result(second)
    finally:
        await aclose(lex)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "serendipity"))
