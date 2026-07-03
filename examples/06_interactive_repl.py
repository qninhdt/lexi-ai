"""Trial 06 — interactive REPL: type words, watch them cache.

A tiny loop so you can probe the dictionary by hand. First lookup of a word
hits the LLM; repeat lookups are instant cache reads. The DB persists between
runs (``LEXI_DB_URL``), so words you looked up earlier stay free.

    uv run python examples/06_interactive_repl.py

Commands:  a word/phrase to look it up  ·  :q or Ctrl-D to quit
"""

import asyncio
import time

from _common import aclose, build_lexicon, lookup, print_result


async def main() -> None:
    lex = build_lexicon()
    await lex.init()
    print("lexi-ai REPL — type a word/phrase, ':q' to quit.\n")
    try:
        while True:
            try:
                word = (await asyncio.to_thread(input, "lookup> ")).strip()
            except EOFError:
                print()
                break
            if not word:
                continue
            if word in {":q", ":quit", ":exit"}:
                break
            t0 = time.perf_counter()
            try:
                result = await lookup(lex, word)
            except Exception as exc:  # noqa: BLE001 - REPL: show, don't crash
                print(f"  ! error: {exc}\n")
                continue
            print(f"[{time.perf_counter() - t0:.2f}s]")
            print_result(result)
            print()
    finally:
        await aclose(lex)


if __name__ == "__main__":
    asyncio.run(main())
