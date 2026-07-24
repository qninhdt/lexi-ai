"""Conformance: the correct answer is never reachable on a presentation type.

Uses a POSITIVE per-variant field allowlist (not just a name blocklist) so a
renamed answer field cannot slip through, plus a name-heuristic as a second net.
Also asserts the internal ``GradingSpec`` family is not importable from the public
``lexi_ai.contracts`` package.
"""

import dataclasses
import importlib
import typing

from lexi_ai.contracts import questions as q

# Every field a presentation type is ALLOWED to expose. Anything else fails.
PRESENTATION_ALLOWLIST: dict[type, set[str]] = {
    q.SingleChoice: {"stem", "options"},
    q.TextSpan: {"stem_with_blank", "word_bank"},
    q.FreeText: {"prompt"},
    q.Flashcard: {"word", "definition", "pos", "example", "ipa_uk", "ipa_us"},
    q.PresentedQuestion: {
        "question_id",
        "type_id",
        "interaction",
        "difficulty_level",
        "render",
        "sense_id",
        "word_id",
    },
}

# Substrings that betray an answer key leaking onto a presentation type.
_BANNED = ("correct", "answer", "rubric", "target", "is_correct", "accepted_forms", "reveal")


def test_presentation_types_only_expose_allowlisted_fields():
    for cls, allowed in PRESENTATION_ALLOWLIST.items():
        names = {f.name for f in dataclasses.fields(cls)}
        extra = names - allowed
        assert not extra, f"{cls.__name__} exposes unexpected field(s): {sorted(extra)}"


def test_every_render_contract_variant_is_allowlisted():
    # A future render variant added to the union MUST get an allowlist entry,
    # forcing an answer-safety review instead of silently going untested.
    for variant in typing.get_args(q.RenderContract):
        assert variant in PRESENTATION_ALLOWLIST, (
            f"RenderContract variant {variant.__name__} has no allowlist entry — "
            f"add one and confirm it exposes no answer"
        )


def test_no_grading_field_names_on_presentation_types():
    for cls in PRESENTATION_ALLOWLIST:
        for f in dataclasses.fields(cls):
            low = f.name.lower()
            assert not any(b in low for b in _BANNED), (
                f"{cls.__name__}.{f.name} looks like a grading field on a "
                f"presentation type"
            )


def test_grading_spec_not_public_in_contracts():
    contracts = importlib.import_module("lexi_ai.contracts")
    for name in (
        "GradingSpec",
        "ChoiceGrading",
        "SpanGrading",
        "RubricGrading",
        "StoredQuestion",
    ):
        assert not hasattr(contracts, name), (
            f"{name} must NOT be public in lexi_ai.contracts"
        )
    # It lives only in the internal domain module.
    dom = importlib.import_module("lexi_ai.domain.questions")
    assert hasattr(dom, "GradingSpec")
    assert hasattr(dom, "StoredQuestion")


def test_stored_question_presentation_is_answer_free():
    # StoredQuestion binds a PresentedQuestion (answer-free) + a separate spec;
    # the presentation half must be the exact answer-free contract type.
    from lexi_ai.domain.questions import StoredQuestion

    ann = {f.name: f.type for f in dataclasses.fields(StoredQuestion)}
    assert "presentation" in ann and "grading" in ann
    # grading is a distinct attribute, never merged into presentation.
    assert "grading" not in {f.name for f in dataclasses.fields(q.PresentedQuestion)}
