"""Deterministic level-0 exposure card."""

from lexi_ai.contracts.questions import QuestionTypeInfo, RenderKind
from lexi_ai.domain.questions import PersistedQuestion
from lexi_ai.questions.base import (
    QuestionContext,
    QuestionQuery,
    register,
)
from lexi_ai.questions.types._shared import _exposure_question


class Flashcard:
    info = QuestionTypeInfo(
        type_id="flashcard",
        render_kind=RenderKind.FLASHCARD,
        interaction="exposure",
        difficulty_levels=frozenset({0}),
    )

    async def retrieve(self, ctx: QuestionContext, query: QuestionQuery) -> PersistedQuestion:
        if query.difficulty_level != 0:
            raise ValueError("flashcard retrieval requires difficulty level 0")
        if ctx.sense_loader is None:
            raise RuntimeError("flashcard retrieval requires ctx.sense_loader")
        entry = await ctx.sense_loader.load_entry(query.sense_id)
        if entry is None:
            raise LookupError(f"sense {query.sense_id} was not found")
        sense = next(
            (item for item in entry.senses if item.sense_id == query.sense_id),
            None,
        )
        if sense is None:
            raise LookupError(f"sense {query.sense_id} was not found in its entry")
        return _exposure_question(entry, sense)


register(Flashcard)
