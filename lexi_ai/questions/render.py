"""Typed projections from a stored flat payload to the answer-safe boundary.

Single source of truth, keyed by :class:`~lexi_ai.contracts.questions.RenderKind`,
that decides three things about a stored ``payload`` dict:

* :func:`to_render` — the learner-visible ``RenderContract`` (answer-free).
* :func:`to_grading` — the internal :class:`GradingSpec` (the answer key; never
  presented). ``None`` for exposure (flashcard) types, which are never graded.
* :func:`to_reveal` — the post-grading ``Reveal`` (MAY carry the answer). ``None``
  for exposure types.

This is the projection layer that lets the DB payload stay flat (and its
``content_hash`` unchanged) while the public boundary emits only contract types.
"""

from __future__ import annotations

from lexi_ai.contracts.questions import (
    ChoiceReveal,
    Flashcard,
    FreeText,
    PresentedQuestion,
    RenderContract,
    RenderKind,
    Reveal,
    RubricReveal,
    SingleChoice,
    SpanReveal,
    TextSpan,
)
from lexi_ai.domain.questions import (
    ChoiceGrading,
    GradingSpec,
    PersistedQuestion,
    RubricGrading,
    SpanGrading,
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
    """Project a flat payload into its internal grading key, or ``None`` for
    exposure types (a flashcard is never graded)."""
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

    ``None`` for exposure types. For free text the disclosure is a
    ``RubricReveal`` whose ``feedback`` is filled by the grader (the payload alone
    carries no learner-facing feedback).
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

    Ids are stringified. A persisted assessment read always carries an int row id;
    a non-persisted exposure card (built fresh, no row) is keyed by its sense —
    there is exactly one exposure card per sense — so it still gets a stable,
    non-null id. An assessment DRAFT (``question_id`` None) is internal only and
    must never reach here: presenting it would emit ``str(None)``.
    """
    if pq.question_id is not None:
        question_id = str(pq.question_id)
    elif pq.interaction == "exposure" and pq.sense_id is not None:
        question_id = f"exposure:{pq.sense_id}"
    else:
        raise ValueError("cannot present a draft question without a persisted id")
    return PresentedQuestion(
        question_id=question_id,
        type_id=pq.type_id,
        interaction=pq.interaction,
        difficulty_level=pq.difficulty_level,
        render=to_render(pq.render_kind, pq.payload),
        sense_id=str(pq.sense_id) if pq.sense_id is not None else None,
        word_id=str(pq.word_id) if pq.word_id is not None else None,
    )


__all__ = ["to_grading", "to_presented", "to_render", "to_reveal"]
