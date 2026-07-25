"""Trial 04 — concurrency: N simultaneous misses generate exactly once.

Fires several lookups (search -> generate the top hit) for the same never-seen
word at once. The per-key asyncio lock plus the in-lock DB double-check
(api.Lexicon) collapse them to a single LLM generation; the rest wait and read
the freshly cached row.

We count real generations by wrapping the generator's ``generate`` method.

    uv run python examples/04_concurrent_lookups.py [word] [n]
"""

import asyncio
import sys

from _common import aclose, build_lexicon, lookup, print_result


async def main(word: str, n: int) -> None:
    lex = build_lexicon()
    await lex.init()

    # Count actual LLM generations by wrapping the bound generator. Forward every
    # argument untouched: this wrapper must not restate the generator's signature,
    # or it silently breaks the next time a parameter is added (it did — a later
    # `existing_tags` argument made this example raise TypeError).
    calls = 0
    original = lex._providers.generator.generate

    async def counting_generate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return await original(*args, **kwargs)

    lex._providers.generator.generate = counting_generate  # type: ignore[method-assign]

    try:
        print(f"=== {n} concurrent lookups of {word!r} (all miss at once) ===")
        results = await asyncio.gather(*[lookup(lex, word) for _ in range(n)])

        entries = [r for r in results if r is not None]
        norms = {e.norm for e in entries}
        print(f"returned {len(results)} results, {len(entries)} entries, distinct norms: {norms}")
        print(f"actual LLM generations: {calls}  (expected 1)")
        print()
        if entries:
            print_result(entries[0])
    finally:
        await aclose(lex)


if __name__ == "__main__":
    word = sys.argv[1] if len(sys.argv) > 1 else "ephemeral"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    asyncio.run(main(word, n))
