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


class GeneratedTheme(BaseModel):
    name: str = Field(
        description="A creative and catchy display name for the theme (e.g., 'The Salty Pirate Captain')."
    )
    description: str = Field(
        description="A short creative description explaining the style's vibe and context (1-2 sentences)."
    )
    style_prompt: str = Field(
        description=(
            "An expanded, highly detailed set of system instructions for other LLMs. "
            "It must clearly guide them on how to restyle dictionary definitions and write example sentences in this specific voice."
        )
    )
    emoji: str = Field(
        description="A single representative emoji (e.g., '🏴‍☠️')."
    )
    tone: list[str] = Field(
        description="A list of 2-4 tone adjectives representing this style (e.g., ['adventurous', 'salty', 'archaic'])."
    )
