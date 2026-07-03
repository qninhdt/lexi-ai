"""Trial 03 — placeholders: natural phrasing resolves to the brace-token norm.

Idioms/phrasal verbs are stored canonically with placeholder tokens
(``{sb}``, ``{sth}``). ``match_key`` folds "somebody"/"someone"/"sb" (and
diacritics, case, spacing) to the same key, so a learner can type the phrase
however they like and still hit one cached entry. ``display`` expands the
tokens back to real words.

    uv run python examples/03_placeholder_lookup.py "look after somebody"
"""

import asyncio
import sys

from _common import aclose, build_lexicon, lookup, print_result

from lexi_ai.normalize import match_key, render


async def main(phrase: str) -> None:
    lex = build_lexicon()
    await lex.init()
    try:
        print(f"=== Lookup: {phrase!r} ===")
        print(f"    match_key -> {match_key(phrase)!r}")
        entry = await lookup(lex, phrase)
        print_result(entry)

        if entry is not None:
            print(f"\n    stored norm : {entry.norm!r}")
            print(f"    display     : {entry.display!r}  (render -> {render(entry.norm)!r})")

            # A differently-phrased variant of the SAME idea should hit the cache.
            variant = phrase.replace("somebody", "someone").replace("something", "sth")
            if variant != phrase:
                print(f"\n=== Variant phrasing: {variant!r} (same match_key -> cache hit) ===")
                print(
                    f"    match_key -> {match_key(variant)!r}  "
                    f"(== original: {match_key(variant) == match_key(phrase)})"
                )
                again = await lookup(lex, variant)
                if again is not None:
                    print(
                        f"    resolved to norm={again.norm!r}  "
                        f"(same entry: {again.norm == entry.norm})"
                    )
    finally:
        await aclose(lex)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "look after somebody"))
