"""Shared helpers for the live-API trials.

Builds a :class:`~lexi_ai.api.Lexicon` wired to the OpenAI-compatible proxy
configured in ``.env`` (see ``.env.example``).

Why not just ``Lexicon.from_settings()``?
    The library's DEFAULT structured-output method is ``json_schema`` (the SDK's
    strict native parse). The proxy used here does NOT enforce that strictly — it
    returns loose JSON that fails validation (missing required fields, wrong enum
    casing). The library seam now supports a ``function_calling`` method (a forced
    single tool) which the proxy honors reliably; we select it here via
    ``LEXI_LLM_STRUCTURED_METHOD=function_calling`` in ``.env`` (or the default
    set below). No LangChain — everything rides the library's own OpenAI seam.
    The MODEL is never hardcoded: it comes from ``LEXI_LLM_MODEL``.

Run every script from the repo root so ``.env`` and ``./data`` resolve:

    uv run python examples/01_lookup_word.py
"""

from __future__ import annotations

from lexi_ai.api import Lexicon
from lexi_ai.config import Settings
from lexi_ai.db import create_engine, create_session_factory
from lexi_ai.embeddings import Embedder
from lexi_ai.generation.generator import Generator
from lexi_ai.llm import build_structured_llm
from lexi_ai.read_models import Entry, SearchResult
from lexi_ai.references.cambridge import CambridgeSource
from lexi_ai.references.loader import ReferenceLoader
from lexi_ai.references.wordnet import WordNetSource


def get_demo_settings() -> Settings:
    """Load settings, defaulting to the proxy-friendly ``function_calling`` method.

    The library default is ``json_schema`` (strict, best on real OpenAI). The demo
    proxy needs ``function_calling``; ``.env`` can override either knob
    (``LEXI_LLM_STRUCTURED_METHOD`` / ``LEXI_LLM_REASONING_EFFORT``).
    """
    settings = Settings()
    if not settings.llm_structured_method:
        settings.llm_structured_method = "function_calling"
    return settings


def build_lexicon(settings: Settings | None = None) -> Lexicon:
    """Assemble a Lexicon with a proxy-compatible generator + local embedder.

    The generator's structured LLM comes straight from the library seam
    (:func:`lexi_ai.llm.build_structured_llm`), reading method/reasoning/model
    from settings — no LangChain. The embedder is the library default (local
    ``transformers``, model from ``LEXI_EMBEDDING_MODEL``); if the
    ``[embeddings]`` extra isn't installed, generation still works and embeddings
    are skipped best-effort — see ``examples/09_semantic_search.py``.
    """
    settings = settings or get_demo_settings()
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    generator = Generator(structured_llm=build_structured_llm(settings), settings=settings)
    loader = ReferenceLoader(CambridgeSource(settings.cambridge_db_path), WordNetSource())
    embedder = Embedder(settings=settings)
    return Lexicon(session_factory, loader, generator, engine=engine, embedder=embedder)


async def aclose(lex: Lexicon) -> None:
    """Dispose the engine so the event loop shuts down cleanly."""
    if lex._engine is not None:
        await lex._engine.dispose()


async def lookup(lex: Lexicon, raw: str) -> Entry | None:
    """Convenience for the simple demos: search a raw string and create the top hit.

    Real callers do this in two explicit steps (``search`` then ``generate`` on a
    chosen result). This wraps "take the best match" so a one-word demo stays short:
    returns the created/loaded :class:`Entry`, or ``None`` when nothing matches
    (the caller can then fall back to ``generate(raw)`` for a custom entry).
    """
    results = await lex.search(raw)
    if not results:
        return None
    return await lex.generate(results[0])


# --- pretty printers ------------------------------------------------------


def print_results(results: list[SearchResult]) -> None:
    """Print a ranked search list: generated hits vs generatable suggestions."""
    if not results:
        print("◇ no matches (not in Cambridge — use generate(raw) for a custom entry)")
        return
    print(f"◇ {len(results)} results (best first):")
    for r in results:
        if r.generated:
            tag = f"✓ generated  id={r.lexi_word_id}"
        else:
            tag = f"+ suggestion cam={r.cambridge_id}"
        gloss = f"  — {r.gloss[:46]}" if r.gloss else ""
        print(f"   [{r.score:.2f}] {r.display}  ({r.entry_type}) {tag}{gloss}")


def print_result(result: Entry | None) -> None:
    """Print an Entry, or a not-found notice."""
    if result is None:
        print("◇ nothing generated (no match)")
    else:
        print_entry(result)


def print_entry(entry: Entry) -> None:
    print(f"● {entry.display}   [{entry.entry_type} | pos={entry.pos} | status={entry.status}]")
    if entry.norm != entry.display:
        print(f"  norm: {entry.norm}")
    print(f"  senses ({len(entry.senses)}):")
    for i, s in enumerate(entry.senses, 1):
        meta = " ".join(x for x in (s.pos, s.cefr_level, s.tier) if x)
        print(f"   {i}. ({meta}) {s.definition}")
        for ex in s.examples[:2]:
            print(f"        · {ex}")
        if s.references:
            refs = ", ".join(f"{r.source}:{r.source_ref}" for r in s.references)
            print(f"        refs: {refs}")
    if entry.aliases:
        print("  aliases: " + ", ".join(f"{a.display} ({a.type})" for a in entry.aliases))
    if entry.links:
        print("  related: " + ", ".join(f"{ln.display} ({ln.rel_type})" for ln in entry.links))


def print_hits(hits: list) -> None:
    """Print semantic-search hits (SemanticHit), best first."""
    if not hits:
        print("◇ no semantic hits (nothing embedded yet — see the note below)")
        return
    print(f"◇ {len(hits)} semantic hits (by meaning, best first):")
    for h in hits:
        print(f"   [{h.score:.3f}] {h.display}  ({h.entry_type}) — {h.sense.definition[:56]}")
