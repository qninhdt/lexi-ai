"""Pydantic schemas for the questions subsystem.

Two kinds live here:

* **LLM structured-output** — ``GeneratedMCQ`` is the response schema the one llm
  plugin (``contextual_mcq``) passes to ``StructuredLLM.parse``, mirroring
  ``generation/schemas.py`` (bounded fields, injectable, retried). ``Judgment``
  (added with the scorers) is the llm-judge output for rubric grading.
* **Payload validators** — one model per type validates the ``payload`` dict a
  plugin builds BEFORE it is persisted, so a bad index or empty option can never
  reach the persistence boundary. The validated model is dumped to the stored
  flat payload; the answer-free presentation and the internal grading spec are
  projected from it by :mod:`lexi_ai.questions.render`.
"""

from pydantic import BaseModel, Field, model_validator

# --- LLM structured output ------------------------------------------------


class GeneratedMCQ(BaseModel):
    """Structured output for the llm contextual-MCQ generator."""

    stem: str = Field(
        max_length=512,
        description="Novel sentence/context with the target sense implied.",
    )
    correct: str = Field(
        max_length=128,
        description="The correct answer (the target word's display).",
    )
    distractors: list[str] = Field(
        min_length=2,
        max_length=3,
        description="Plausible wrong options.",
    )


class Judgment(BaseModel):
    """Structured output for the llm rubric judge (free-text grading)."""

    correct: bool = Field(description="Does the answer satisfy the rubric?")
    score: float = Field(ge=0.0, le=1.0, description="0.0..1.0 quality/partial credit.")
    feedback: str = Field(max_length=512, description="Short learner-facing feedback.")


# --- payload validators (one per render format) ---------------------------


class FlashcardPayload(BaseModel):
    """Deterministic level-0 exposure card built from authoritative sense data."""

    word: str = Field(min_length=1, max_length=128)
    pos: str | None = Field(default=None, max_length=32)
    definition: str = Field(min_length=1, max_length=2048)
    example: str | None = Field(default=None, max_length=512)
    ipa_uk: str | None = Field(default=None, max_length=128)
    ipa_us: str | None = Field(default=None, max_length=128)


class MCQPayload(BaseModel):
    """Payload for single-choice formats (``definition_mcq``, ``contextual_mcq``)."""

    stem: str = Field(min_length=1, max_length=512)
    options: list[str] = Field(min_length=2)
    correct_index: int = Field(ge=0)

    @model_validator(mode="after")
    def _index_in_range(self) -> "MCQPayload":
        if self.correct_index >= len(self.options):
            raise ValueError("correct_index out of range for options")
        return self


class ClozePayload(BaseModel):
    """Payload for the ``cloze`` text-span format.

    ``accepted_forms`` are extra surfaces the grader folds equal to the answer
    (the sense's inflected forms), so a learner typing ``ran`` for ``run`` scores
    right without touching ``match_key``. Empty when the sense has no forms."""

    stem_with_blank: str = Field(min_length=1, max_length=512)
    answer_norm: str = Field(min_length=1, max_length=128)
    accepted_forms: list[str] = Field(default_factory=list)
    word_bank: list[str] = Field(default_factory=list)


class UseInSentencePayload(BaseModel):
    """Payload for the ``use_in_sentence`` free-text format."""

    prompt: str = Field(min_length=1, max_length=512)
    target_norm: str = Field(min_length=1, max_length=128)
    rubric: str = Field(min_length=1, max_length=512)
