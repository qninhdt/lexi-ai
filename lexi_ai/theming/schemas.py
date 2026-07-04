"""Structured-output schema for themed generation (Phase 2).

The model returns ``ThemedResult.senses`` in the SAME order the neutral senses
were numbered in the prompt: index ``i`` maps to the i-th neutral sense's id.
No free-form sense id from the LLM (avoids a hallucinated-id path) — the api
layer supplies the ordered ``sense_ids`` and the persistence layer zips by
position under a hard length guard.
"""

from pydantic import BaseModel, Field


class ThemedSense(BaseModel):
    definition: str = Field(description="The sense's definition, restyled in the theme's voice.")
    examples: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Fresh in-voice example sentences illustrating this sense.",
    )


class ThemedResult(BaseModel):
    senses: list[ThemedSense] = Field(
        min_length=1,
        description=(
            "One themed sense per numbered neutral sense, IN THE SAME ORDER. "
            "Return exactly as many senses as were given, no more, no fewer."
        ),
    )
