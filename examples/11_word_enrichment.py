"""Trial 11 — word-content enrichment (learner-dictionary fields).

Every generated word now carries seven learner-dictionary enrichments, all
emitted in the SAME LLM call as its senses (no extra requests) and anchored to
Cambridge/WordNet for hallucination control — we SYNTHESIZE our own data, we do
not copy the source. They split by one test: does the field NAME a lemma or
LABEL this sense?

  * word-references (NAME a lemma -> normalized to real entries, like synonyms):
      - word_family   derivational/inflectional relatives (happy -> happiness)
      - confused_with commonly-confused near-forms (affect <-> effect)
    Both surface in ``entry.links`` by ``rel_type`` (normalized via match_key).

  * sense labels (LABEL this sense -> columns / a child table):
      - guideword     short homograph disambiguator (bank -> MONEY vs RIVER)
      - grammar       0-3 closed-vocab labels (countable, transitive, ...)
      - register      formality band (formal / informal / slang / ...)
      - connotation   affective tone (positive / negative / neutral)
      - collocations  high-frequency partner phrases (make a decision)

    uv run python examples/11_word_enrichment.py [word]

Like the sibling trials, the FIRST lookup of a word spends tokens; re-runs read
the local cache. Needs the LLM env (LEXI_LLM_*); prints a notice and exits if a
word can't be resolved.
"""

import asyncio
import sys

from _common import aclose, build_lexicon, lookup

DEFAULT_WORD = "bank"


def _print_enrichment(entry) -> None:
    """Print the per-sense labels + collocations and the word-reference links."""
    print(f"● {entry.display}   [{entry.entry_type} | pos={entry.pos} | status={entry.status}]")
    print(f"  senses ({len(entry.senses)}):")
    for i, s in enumerate(entry.senses, 1):
        meta = " ".join(x for x in (s.pos, s.cefr_level, s.tier) if x)
        guide = f"  «{s.guideword}»" if s.guideword else ""
        print(f"   {i}. ({meta}){guide} {s.definition}")
        labels = []
        if s.grammar:
            labels.append("grammar: " + ", ".join(s.grammar))
        if s.register:
            labels.append(f"register: {s.register}")
        if s.connotation:
            labels.append(f"connotation: {s.connotation}")
        if labels:
            print("        " + " | ".join(labels))
        if s.collocations:
            print("        collocations: " + "; ".join(s.collocations))

    # word_family / confused_with ride the normalized links path (by rel_type).
    fam = [ln.display for ln in entry.links if ln.rel_type == "word_family"]
    conf = [ln.display for ln in entry.links if ln.rel_type == "confused_with"]
    if fam:
        print("  word family: " + ", ".join(fam))
    if conf:
        print("  confused with: " + ", ".join(conf))
    if not (fam or conf):
        print("  (no word-references proposed for this word)")


async def main() -> None:
    word = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WORD
    lex = build_lexicon()
    await lex.init()
    try:
        print(f"=== generating {word!r} (enrichments ride the same LLM call as senses) ===")
        entry = await lookup(lex, word)
        if entry is None:
            print(f"◇ {word!r} not in Cambridge — try another word, or use a custom add.")
            return
        print()
        _print_enrichment(entry)
    finally:
        await aclose(lex)


if __name__ == "__main__":
    asyncio.run(main())
