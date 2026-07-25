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

If the ``[embeddings]`` extra is not installed, generation still works (vectors
are best-effort on the write path) but ``semantic_search`` RAISES
``EmbeddingUnavailable`` rather than answering "no match" for a search it never
ran. Handling that named exception — as this script does — is the intended way to
degrade; an empty list always means the search ran and matched nothing.
"""

import asyncio

from _common import aclose, build_lexicon, lookup, print_hits

from lexi_ai.embeddings import EmbeddingUnavailable

_INSTALL_HINT = (
    "\nSemantic search needs the local encoder. Install it with:\n"
    "    uv sync --extra embeddings --extra lancedb\n"
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
        except EmbeddingUnavailable:
            print(_INSTALL_HINT)
    finally:
        await aclose(lex)


if __name__ == "__main__":
    asyncio.run(main())
