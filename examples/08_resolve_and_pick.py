"""Trial 08 — the real flow: search → pick → generate.

A raw string is ambiguous, so you never generate from it directly. You:
  1. search(query)   -> a ranked list of SearchResults (FREE, never generates);
                        each is either generated (lexi_word_id) or a suggestion
                        (cambridge_id, generatable)
  2. inspect / pick one (item 0 is the best match)
  3. generate(result) -> the Entry (generates once, then cached)

Also shows get_status() (FREE) and generate(raw) for a word Cambridge lacks.

    uv run python examples/08_resolve_and_pick.py [query]
"""

import asyncio
import sys

from _common import aclose, build_lexicon, print_entry, print_results


async def main(query: str) -> None:
    lex = build_lexicon()
    await lex.init()
    try:
        # 1. search — FREE, never calls the LLM.
        print(f"=== search({query!r}) — ranked matches (FREE) ===")
        results = await lex.search(query)
        print_results(results)

        if not results:
            print(f"\nNothing in Cambridge. Creating a custom entry via generate({query!r})…")
            entry = await lex.generate(query)
            print_entry(entry)
            return

        # 2. pick the best match.
        best = results[0]
        kind = "already generated" if best.generated else "a suggestion (generatable)"
        print(f"\npicked: {best.display!r} — {kind}")

        # 3. introspect before generating — FREE.
        if best.generated:
            print(f"  status: {await lex.get_status(best.lexi_word_id)!r}")
        else:
            print("  status: not generated yet (no lexi_word_id)")

        # 4. generate (or return cached).
        print(f"\n=== generate(result) — create or return {best.display!r} ===")
        entry = await lex.generate(best)
        print_entry(entry)

        # 5. now it's cached — search again folds it into a generated hit.
        again = await lex.search(query)
        top = again[0]
        print(
            f"\nafter generation, top result: generated={top.generated} "
            f"id={top.lexi_word_id} (was generated={best.generated})"
        )
    finally:
        await aclose(lex)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "serendipity"))
