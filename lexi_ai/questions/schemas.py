"""Pydantic schemas for the questions subsystem.

Two kinds live here:

* **LLM structured-output** — ``GeneratedMCQ`` is the response schema the one llm
  plugin (``contextual_mcq``) passes to ``StructuredLLM.parse``, mirroring
  ``generation/schemas.py`` (bounded fields, injectable, retried). ``Judgment``
  (added with the scorers) is the llm-judge output for rubric grading.
* **Payload validators** — one model per format validates the ``payload`` dict a
  plugin builds BEFORE it becomes a :class:`~lexi_ai.read_models.Question`, so a
  bad index or empty option can never reach the persistence boundary. The
  validated model is dumped straight to ``Question.payload`` (a plain dict).
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


class MatchingPayload(BaseModel):
    """Payload for the ``matching`` format: pair each left cue with a right item.

    ``lefts`` are the cues (guidewords) in stable order; ``rights`` are the
    definitions in a shuffled order; ``correct_map[i]`` is the index into
    ``rights`` that the i-th left pairs with. An answer is a list of the same
    length assigning each left a right-index; grading compares it to ``correct_map``.
    """

    prompt: str = Field(min_length=1, max_length=512)
    lefts: list[str] = Field(min_length=2)
    rights: list[str] = Field(min_length=2)
    correct_map: list[int] = Field(min_length=2)

    @model_validator(mode="after")
    def _shapes_agree(self) -> "MatchingPayload":
        if not (len(self.lefts) == len(self.rights) == len(self.correct_map)):
            raise ValueError("lefts, rights, and correct_map must be equal length")
        if sorted(self.correct_map) != list(range(len(self.rights))):
            raise ValueError("correct_map must be a permutation of the right indices")
        return self


class AudioRef(BaseModel):
    """A durable pointer to a TTS clip: the ``(source_kind, source_id, voice, fmt)``
    reference tuple, NOT an ``assets`` row id. A row id dangles after a purge, but
    the reference re-resolves the clip cache-first at play time (Phase 1 verifies
    ``content_hash`` on read, so a regenerated source yields a clean miss — never
    stale audio). Frozen into the question payload as plain JSON."""

    source_kind: str = Field(min_length=1, max_length=32)
    source_id: int = Field(ge=0)
    voice: str = Field(min_length=1, max_length=64)
    fmt: str = Field(min_length=1, max_length=16)


class ListeningPayload(BaseModel):
    """Payload for the ``listening`` single-choice format: hear a clip, pick the word.

    ``audio_ref`` addresses the clip by reference tuple; grading is index-based
    (:func:`grade_single_choice`) and never touches the clip, so a dangling ref
    still grades."""

    prompt: str = Field(min_length=1, max_length=512)
    audio_ref: AudioRef
    options: list[str] = Field(min_length=2)
    correct_index: int = Field(ge=0)

    @model_validator(mode="after")
    def _index_in_range(self) -> "ListeningPayload":
        if self.correct_index >= len(self.options):
            raise ValueError("correct_index out of range for options")
        return self


class SpellingPayload(BaseModel):
    """Payload for the ``spelling`` text-span format: hear a clip, type the word.

    ``audio_ref`` addresses the clip by reference tuple; grading is ``match_key``
    equality (:func:`grade_text_span`) against ``answer_norm`` and never touches
    the clip, so a dangling ref still grades."""

    prompt: str = Field(min_length=1, max_length=512)
    audio_ref: AudioRef
    answer_norm: str = Field(min_length=1, max_length=128)
    accepted_forms: list[str] = Field(default_factory=list)
