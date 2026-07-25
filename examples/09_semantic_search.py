"""Trial 09 — semantic search over generated senses (local embeddings).

Every sense gets an embedding vector at generation time (best-effort, computed
LOCALLY by a transformers model — the chat proxy has no embeddings endpoint). The
vectors live in a dedicated index, NOT in the dictionary database.
``semantic_search`` then ranks already-generated senses by meaning, not spelling.

    # one-time: install the encoder and the durable vector index
    uv sync --extra embeddings --extra lancedb
    uv run python examples/09_semantic_search.py

Flow:
  1. generate a handful of words (each sense is embedded on the way in)
  2. semantic_search(query) -> senses closest in MEANING to the query (FREE)
  3. backfill_embeddings() -> fills any vectors skipped when the extra was absent

Semantic search is OFF by default, so this example needs it switched on:

    uv sync --extra embeddings --extra lancedb
    LEXI_VECTOR_BACKEND=lancedb uv run python examples/09_semantic_search.py

Whenever the feature cannot run — switched off, index extra missing, encoder extra
missing — ``semantic_search`` RAISES rather than answering "no match" for a search
it never ran. Catching ``SemanticSearchUnavailable`` (the base class of all three)
is the intended way to degrade; an empty list always means the search ran and
matched nothing. Generation is unaffected either way: it just stores no vectors.
"""

import asyncio

from _common import aclose, build_lexicon, lookup, print_hits

from lexi_ai.domain.errors import SemanticSearchUnavailable

# Catch the BASE class, not one subclass: the feature has three independent ways to
# be unavailable and an example that named just one would crash on the other two.
_ENABLE_HINT = (
    "\nSemantic search is off or incomplete. Enable it with:\n"
    "    uv sync --extra embeddings --extra lancedb\n"
    "    export LEXI_VECTOR_BACKEND=lancedb\n"
    "then re-run — backfill_embeddings() will vectorize the senses generated above."
)

SEED_WORDS = ["serendipity", "meticulous", "car", "happy", "melancholy"]
QUERIES = [
    "a chance lucky discovery",
    "very careful and precise about details",
    "a feeling of sadness",
]


async def main() -> None:
    lex = build_lexicon()
    await lex.init()
    try:
        # 1. Generate a few words. Each sense is embedded best-effort as it lands.
        print("=== generating seed words (senses embedded on the way in) ===")
        for word in SEED_WORDS:
            entry = await lookup(lex, word)
            made = entry.display if entry else "(no match)"
            print(f"  · {word:<12} -> {made}")

        try:
            # 2. Fill any vectors that were skipped (e.g. extra installed just now).
            filled = await lex.engine().backfill_embeddings()
            if filled:
                print(f"\nbackfill_embeddings(): embedded {filled} sense(s)")

            # 3. Search by meaning — FREE, never generates a dictionary entry.
            for query in QUERIES:
                print(f"\n=== semantic_search({query!r}) ===")
                # An empty result here means no sense was close enough — the
                # encoder and index both worked.
                print_hits(await lex.reader().semantic_search(query, k=5))
        except SemanticSearchUnavailable as exc:
            print(f"\n{type(exc).__name__}: {exc}")
            print(_ENABLE_HINT)
    finally:
        await aclose(lex)


if __name__ == "__main__":
    asyncio.run(main())
