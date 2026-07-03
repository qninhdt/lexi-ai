"""Trial 07 — inspect matching: how a string input maps to Cambridge + WordNet.

Two layers turn a raw string into anchored senses:

  Layer 1 — ANCHOR SELECTION (deterministic, no LLM). The loader picks which
    Cambridge `words` row and which WordNet synsets to pull:
      * Cambridge: exact match on slug/display_form, else a `match_key` fallback
        (lowercase + strip diacritics + fold placeholders + collapse spaces).
      * WordNet: looked up on Cambridge's display_form (POS-filtered) when
        Cambridge hit, else on the raw input; NLTK morphy folds inflections
        (running -> run).

  Layer 2 — SENSE MAPPING (LLM). Each synthesized sense carries `references`
    pointing back to a specific Cambridge sense id / WordNet synset key. This
    reads them from the cached entry and resolves each ref to its anchor.

    uv run python examples/07_inspect_matching.py [word]
"""

import asyncio
import sys

from _common import aclose, build_lexicon, lookup

from lexi_ai.constants import canonical_cambridge_ref
from lexi_ai.normalize import match_key


async def main(word: str) -> None:
    lex = build_lexicon()
    await lex.init()
    try:
        key = match_key(word)
        print(f"input string : {word!r}")
        print(f"match_key    : {key!r}   (lowercase, no diacritics, folded placeholders)")

        # --- Layer 1: deterministic anchor selection (no LLM) ---------------
        bundle = await lex._loader.bundle(word)

        print("\n--- CAMBRIDGE anchor (deterministic) ---")
        if bundle.cambridge_word_id is None:
            print("  (no Cambridge row matched — will rely on WordNet + general knowledge)")
        else:
            same = match_key(bundle.word_raw) == key
            print(
                f"  matched word_id={bundle.cambridge_word_id}  "
                f"display_form={bundle.word_raw!r}  entry_type={bundle.entry_type}"
            )
            print(
                f"  match_key(display_form)={match_key(bundle.word_raw)!r}  "
                f"== match_key(input) -> {same}"
            )
            print(f"  cambridge senses pulled ({len(bundle.cambridge_senses)}):")
            for s in bundle.cambridge_senses:
                print(
                    f"    sense#{s.cambridge_sense_id} (pos={s.pos}, cefr={s.cefr_level}) "
                    f"{s.definition[:58]}"
                )

        print("\n--- WORDNET anchor (deterministic, via NLTK) ---")
        if not bundle.wordnet_synsets:
            print("  (no synsets — normal for idioms/phrases)")
        else:
            print(
                f"  synsets pulled ({len(bundle.wordnet_synsets)}) looked up on "
                f"{bundle.word_raw!r}:"
            )
            for s in bundle.wordnet_synsets:
                print(f"    {s.name} (pos={s.pos}) {s.definition[:58]}")

        # --- Layer 2: LLM maps each synthesized sense back to an anchor -----
        # Read from the cached entry (generates once if new, then free).
        cam_by_id = {str(s.cambridge_sense_id): s for s in bundle.cambridge_senses}
        wn_by_name = {s.name: s for s in bundle.wordnet_synsets}

        print("\n--- GENERATED senses -> references (LLM mapping) ---")
        entry = await lookup(lex, word)
        if entry is None:
            print("  (no Cambridge match — run on a resolvable word)")
            return
        for i, se in enumerate(entry.senses, 1):
            print(f"  sense {i} ({se.tier} {se.cefr_level}): {se.definition[:58]}")
            for r in se.references:
                if r.source == "cambridge":
                    tgt = cam_by_id.get(canonical_cambridge_ref(r.source_ref))
                    desc = tgt.definition[:48] if tgt else "(id not in this bundle)"
                else:
                    tgt = wn_by_name.get(r.source_ref)
                    desc = tgt.definition[:48] if tgt else "(synset not in this bundle)"
                print(f"      -> {r.source}:{r.source_ref}  = {desc}")
    finally:
        await aclose(lex)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "happy"))
