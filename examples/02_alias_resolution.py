"""Trial 02 — aliases: surface variants resolve to one cached entry.

The generator emits a US-canonical ``norm`` plus a ``spelling_uk`` alias, so a
UK spelling typed later resolves to the SAME word with no new generation. This
looks up the canonical form once, then queries a variant and shows it hits the
same ``norm``.

    uv run python examples/02_alias_resolution.py [canonical] [variant]
    uv run python examples/02_alias_resolution.py color colour
"""

import asyncio
import sys

from _common import aclose, build_lexicon, lookup, print_result


async def main(canonical: str, variant: str) -> None:
    lex = build_lexicon()
    await lex.init()
    try:
        print(f"=== Generate canonical form: {canonical!r} ===")
        entry = await lookup(lex, canonical)
        print_result(entry)

        if entry is not None:
            alias_forms = [a.display for a in entry.aliases]
            print(f"\naliases recorded: {alias_forms or '(none)'}")

        print(f"\n=== Now query the variant: {variant!r} (should hit same entry) ===")
        via_variant = await lookup(lex, variant)
        print_result(via_variant)

        if entry is not None and via_variant is not None:
            same = via_variant.norm == entry.norm
            verdict = "SAME entry (alias resolved)" if same else "DIFFERENT entry"
            print(f"\n-> {variant!r} resolved to norm={via_variant.norm!r}: {verdict}")
    finally:
        await aclose(lex)


if __name__ == "__main__":
    canonical = sys.argv[1] if len(sys.argv) > 1 else "color"
    variant = sys.argv[2] if len(sys.argv) > 2 else "colour"
    asyncio.run(main(canonical, variant))
