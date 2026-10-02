"""Explicit per-call parallel Sense Linking; never invoked by Word generation."""

import asyncio
from dataclasses import dataclass

from lexi_ai.errors import MissingProviderError
from lexi_ai.inference.config import DecisionMode
from lexi_ai.inference.prompting import render_decision
from lexi_ai.vocab import POS_TAGS

from .storage import apply_resolution, pending_relations


@dataclass(frozen=True)
class Resolution:
    edge_id: int
    state: str
    error: str | None = None


async def resolve_relations(
    db, decision_model, batch_size: int = 20, *, mode: DecisionMode = DecisionMode.LLM_FALLBACK
) -> list[Resolution]:
    """Caller coordinates overlapping calls; work runs independently within this one call."""
    mode = DecisionMode(mode)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch size must be positive")
    links = await pending_relations(db, min(batch_size, 50))
    if any(link.candidates for link in links) and decision_model is None:
        raise MissingProviderError("Sense Linking requires a configured decision provider")

    async def one(link):
        try:
            if link.source_definition is None or link.source_pos not in POS_TAGS:
                raise ValueError("relation has no valid neutral source meaning/POS")
            if any(candidate.definition is None for candidate in link.candidates):
                raise ValueError("relation target has an incomplete neutral meaning inventory")
            selected = None
            if link.candidates:
                keys = {
                    f"candidate_{i}": candidate for i, candidate in enumerate(link.candidates, 1)
                }
                result = await decision_model.decide(
                    *render_decision(
                        "relations/prompts/resolve_sense_relations.json",
                        source_word=link.source_word,
                        source_definition=link.source_definition,
                        relation_type=link.relation_type,
                        target_word=link.target_word,
                        target_gloss=link.target_gloss,
                        candidates=[
                            {"index": i, "pos": c.pos, "definition": c.definition}
                            for i, c in enumerate(link.candidates, 1)
                        ],
                    ),
                    mode=mode,
                )
                choice = result.choices["matched_sense"].choice
                if choice != "no_candidate":
                    if choice not in keys:
                        raise ValueError("invalid relation candidate key")
                    selected = keys[choice]
            applied = await apply_resolution(db, link.edge_id, selected, expected=link)
            return Resolution(
                link.edge_id,
                "noop" if not applied else ("resolved" if selected else "unresolvable"),
            )
        except Exception as exc:
            # This is the per-link job boundary: transport and DB failures must
            # not cancel independent links. asyncio cancellation still propagates.
            return Resolution(link.edge_id, "error", str(exc))

    return list(await asyncio.gather(*(one(link) for link in links)))
