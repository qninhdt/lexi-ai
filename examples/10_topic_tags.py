"""Trial 10 — LLM-generated topic tags (open-vocabulary, consistent).

Every generated word gets 1-3 broad topic tags. The LLM invents them (no
predefined list), but consistency is enforced by three layers:

  1. the FULL existing tag vocab is injected into each generation prompt for
     reuse ("reuse these; only create new when none fits");
  2. a deterministic ``tag_key`` (lowercase / singular / diacritic-folded)
     dedups case & plural variants on the write path;
  3. resolve-or-create under a UNIQUE key so the same tag is one row.

Each tag has a short ``name`` slug and a human ``title`` (set once, first-seen).

    uv run python examples/10_topic_tags.py

Flow:
  1. generate a handful of words (topics ride the same LLM call as senses)
  2. list_tags()      -> every topic + its live member count (FREE)
  3. list_entries_by_tag(t)  -> every generated word carrying a topic (FREE)
"""

import asyncio

from _common import aclose, build_lexicon, lookup

SEED_WORDS = ["espresso", "latte", "invoice", "mortgage", "jubilant", "grief"]


async def main() -> None:
    lex = build_lexicon()
    await lex.init()
    try:
        # 1. Generate a few words. Each carries topics assigned by the LLM, kept
        #    consistent by the injected vocab + tag_key dedup.
        print("=== generating seed words (topics assigned on the way in) ===")
        for word in SEED_WORDS:
            entry = await lookup(lex, word)
            if entry is None:
                print(f"  · {word:<12} -> (no match)")
                continue
            topics = ", ".join(t.title for t in entry.topics) or "(none)"
            print(f"  · {word:<12} -> {entry.display:<14} topics: {topics}")

        # 2. Browse the topic index — FREE, never generates.
        print("\n=== list_tags() (topic — title — live member count) ===")
        tags = await lex.list_tags()
        if not tags:
            print("  (no topics yet — the LLM may not have assigned any)")
        for t in tags:
            print(f"   {t.name:<12} {t.title:<22} × {t.count}")

        # 3. Filter words by the most-used topic — FREE, resolved via tag_key so
        #    casing/plural variants of the query all hit.
        if tags:
            top = tags[0]
            print(f"\n=== list_entries_by_tag({top.name!r}) ===")
            for hit in await lex.list_entries_by_tag(top.name):
                print(f"   · {hit.display}  ({hit.entry_type})")
    finally:
        await aclose(lex)


if __name__ == "__main__":
    asyncio.run(main())
