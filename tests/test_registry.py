"""Focused Phase 2 contract tests."""

import pytest

from lexi_ai.constants import RENDER_FORMAT_PAYLOAD
from lexi_ai.questions import base, schemas
from lexi_ai.questions.formats import _shared as shared
from lexi_ai.read_models import Entry, SenseView


class _Assessment:
    descriptor = base.QuestionTypeDescriptor(
        type_id="definition_mcq",
        render_format="single_choice",
        supported_levels=frozenset({1}),
        interaction_mode="assessment",
    )

    async def prepare(self, ctx, demands):
        return base.PrepareReport({})

    async def retrieve(self, ctx, query):
        return None

    async def evaluate(self, ctx, question, answer):
        raise NotImplementedError


def _descriptor(**overrides):
    values = {
        "type_id": "definition_mcq",
        "render_format": "single_choice",
        "supported_levels": frozenset({1}),
        "interaction_mode": "assessment",
    }
    values.update(overrides)
    return base.QuestionTypeDescriptor(**values)


def _plugin(descriptor, *, evaluate=True):
    attrs = {
        "descriptor": descriptor,
        "prepare": lambda self, ctx, demands: None,
        "retrieve": lambda self, ctx, query: None,
    }
    if evaluate:
        attrs["evaluate"] = lambda self, ctx, question, answer: None
    return type("FakeType", (), attrs)


@pytest.fixture(autouse=True)
def _restore_registry():
    original = dict(base.REGISTRY)
    base.REGISTRY.clear()
    yield
    base.REGISTRY.clear()
    base.REGISTRY.update(original)


def test_registers_well_formed_assessment_by_type_id():
    base.register(_Assessment)
    assert isinstance(base.REGISTRY["definition_mcq"], _Assessment)


@pytest.mark.parametrize(
    ("descriptor", "message"),
    [
        (_descriptor(type_id="unknown"), "unknown question type"),
        (_descriptor(render_format="unknown"), "unknown render format"),
        (_descriptor(supported_levels=frozenset()), "supported_levels"),
        (_descriptor(supported_levels=frozenset({5})), "supported_levels"),
        (
            _descriptor(
                type_id="flashcard",
                render_format="single_choice",
                supported_levels=frozenset({0}),
                interaction_mode="exposure",
            ),
            "level 0",
        ),
        (
            _descriptor(
                type_id="flashcard",
                render_format="flashcard",
                supported_levels=frozenset({0, 1}),
                interaction_mode="exposure",
            ),
            "level 0",
        ),
    ],
)
def test_register_rejects_invalid_descriptor(descriptor, message):
    with pytest.raises(ValueError, match=message):
        base.register(_plugin(descriptor))


def test_register_rejects_assessment_without_evaluate():
    with pytest.raises(ValueError, match="evaluate"):
        base.register(_plugin(_descriptor(), evaluate=False))


def test_register_rejects_exposure_with_evaluate():
    descriptor = _descriptor(
        type_id="flashcard",
        render_format="flashcard",
        supported_levels=frozenset({0}),
        interaction_mode="exposure",
    )
    with pytest.raises(ValueError, match="must not define evaluate"):
        base.register(_plugin(descriptor, evaluate=True))


def test_register_rejects_missing_render_payload_mapping(monkeypatch):
    monkeypatch.delitem(RENDER_FORMAT_PAYLOAD, "single_choice")
    with pytest.raises(ValueError, match="payload validator"):
        base.register(_Assessment)


def test_render_payload_mapping_resolves_to_schema_classes():
    assert base.payload_model_for("single_choice") is schemas.MCQPayload
    assert base.payload_model_for("flashcard") is schemas.FlashcardPayload


def _entry() -> tuple[Entry, SenseView]:
    sense = SenseView(
        definition="fluent and persuasive in speech",
        tier="core",
        pos="adjective",
        cefr_level="C1",
        examples=["She gave an eloquent speech."],
        ipa_uk="ˈel.ə.kwənt",
        sense_id=7,
    )
    return (
        Entry(
            display="eloquent",
            norm="eloquent",
            entry_type="word",
            pos="adjective",
            status="done",
            word_id=3,
            senses=[sense],
        ),
        sense,
    )


def test_mcq_builder_stamps_new_question_contract():
    entry, sense = _entry()
    question = shared._mcq_question(
        entry,
        sense,
        "Which word matches?",
        "seed",
        ["terse"],
        type_id="definition_mcq",
        difficulty_level=1,
    )
    assert question.type_id == "definition_mcq"
    assert question.render_format == "single_choice"
    assert question.difficulty_level == 1
    assert question.interaction_mode == "assessment"
    assert not hasattr(question, "format")
    assert not hasattr(question, "answer_kind")


def test_exposure_builder_stamps_flashcard_payload():
    entry, sense = _entry()
    question = shared._exposure_question(entry, sense)
    assert question.type_id == "flashcard"
    assert question.render_format == "flashcard"
    assert question.difficulty_level == 0
    assert question.interaction_mode == "exposure"
    assert question.payload == {
        "word": "eloquent",
        "pos": "adjective",
        "definition": "fluent and persuasive in speech",
        "example": "She gave an eloquent speech.",
        "ipa_uk": "ˈel.ə.kwənt",
        "ipa_us": None,
    }


def test_context_exposes_narrow_sense_loader_without_expanding_question_store():
    assert "sense_loader" in base.QuestionContext.__dataclass_fields__
    assert "retrieve_one" in base.QuestionStore.__dict__
    assert "load_entry" not in base.QuestionStore.__dict__
    assert "load_sense" not in base.QuestionStore.__dict__
