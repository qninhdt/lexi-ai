"""Deterministic level-0 exposure card."""

from lexi_ai.questions.base import (
    QuestionContext,
    QuestionQuery,
    QuestionTypeDescriptor,
    register,
)
from lexi_ai.questions.formats._shared import _exposure_question
from lexi_ai.read_models import Question


class Flashcard:
    descriptor = QuestionTypeDescriptor(
        type_id="flashcard",
        render_format="flashcard",
        supported_levels=frozenset({0}),
        interaction_mode="exposure",
    )

    async def retrieve(self, ctx: QuestionContext, query: QuestionQuery) -> Question:
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
