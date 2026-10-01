"""Typed projections from a stored flat payload to the answer-safe boundary.

Single source of truth, keyed by :class:`~lexi_ai.questions.types.RenderKind`: :func:`to_render`
gives the answer-free learner view, :func:`to_grading` the internal answer key, :func:`to_reveal`
the post-grading disclosure. Keeping the payload flat is what preserves ``content_hash`` and dedup
identity.
"""

from __future__ import annotations

from lexi_ai.questions.types import (
    ChoiceGrading,
    ChoiceReveal,
    Flashcard,
    FreeText,
    GradingSpec,
    PersistedQuestion,
    PresentedQuestion,
    RenderContract,
    RenderKind,
    Reveal,
    RubricGrading,
    RubricReveal,
    SingleChoice,
    SpanGrading,
    SpanReveal,
    TextSpan,
)


def to_render(render_kind: RenderKind, payload: dict) -> RenderContract:
    """Project a flat payload into its answer-free presentation contract."""
    if render_kind is RenderKind.SINGLE_CHOICE:
        return SingleChoice(stem=payload["stem"], options=tuple(payload["options"]))
    if render_kind is RenderKind.TEXT_SPAN:
        return TextSpan(
            stem_with_blank=payload["stem_with_blank"],
            word_bank=tuple(payload.get("word_bank", ())),
        )
    if render_kind is RenderKind.FREE_TEXT:
        return FreeText(prompt=payload["prompt"])
    if render_kind is RenderKind.FLASHCARD:
        return Flashcard(
            word=payload["word"],
            definition=payload["definition"],
            pos=payload.get("pos"),
            example=payload.get("example"),
            ipa_uk=payload.get("ipa_uk"),
            ipa_us=payload.get("ipa_us"),
        )
    raise ValueError(f"unknown render kind: {render_kind!r}")


def to_grading(render_kind: RenderKind, payload: dict) -> GradingSpec | None:
    """Project a flat payload into its internal grading key, or ``None`` for exposure types (a
    flashcard is never graded).
    """
    if render_kind is RenderKind.SINGLE_CHOICE:
        return ChoiceGrading(correct_index=payload["correct_index"])
    if render_kind is RenderKind.TEXT_SPAN:
        return SpanGrading(
            answer_norm=payload["answer_norm"],
            accepted_forms=tuple(payload.get("accepted_forms", ())),
        )
    if render_kind is RenderKind.FREE_TEXT:
        return RubricGrading(target_norm=payload["target_norm"], rubric=payload["rubric"])
    if render_kind is RenderKind.FLASHCARD:
        return None
    raise ValueError(f"unknown render kind: {render_kind!r}")


def to_reveal(render_kind: RenderKind, payload: dict) -> Reveal | None:
    """Project a flat payload into its post-grading answer disclosure.

    ``None`` for exposure types. A free text ``RubricReveal`` carries ``feedback`` filled by the
    grader, since the payload alone has no learner-facing feedback.
    """
    if render_kind is RenderKind.SINGLE_CHOICE:
        idx = payload["correct_index"]
        return ChoiceReveal(correct_index=idx, correct_option=payload["options"][idx])
    if render_kind is RenderKind.TEXT_SPAN:
        return SpanReveal(
            correct_answer=payload["answer_norm"],
            accepted_forms=tuple(payload.get("accepted_forms", ())),
        )
    if render_kind is RenderKind.FREE_TEXT:
        return RubricReveal(feedback=None)
    if render_kind is RenderKind.FLASHCARD:
        return None
    raise ValueError(f"unknown render kind: {render_kind!r}")


def to_presented(pq: PersistedQuestion) -> PresentedQuestion:
    """Project an internal carrier into the answer-free public presentation.

    Persisted assessments carry their integer row id. Exposure cards are not persisted, so their
    owning sense id is their stable integer presentation id. An assessment DRAFT (``question_id``
    None) must never reach here.
    """
    if pq.question_id is not None:
        question_id = pq.question_id
    if pq.question_id is None:
        if pq.interaction != "exposure" or pq.sense_id is None:
            raise ValueError("cannot present a draft question without a persisted id")
        question_id = pq.sense_id
    return PresentedQuestion(
        question_id=question_id,
        type_id=pq.type_id,
        interaction=pq.interaction,
        render_kind=pq.render_kind,
        difficulty_level=pq.difficulty_level,
        render=to_render(pq.render_kind, pq.payload),
        sense_id=pq.sense_id,
        word_id=pq.word_id,
    )


__all__ = ["to_grading", "to_presented", "to_render", "to_reveal"]
