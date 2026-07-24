"""Focused contract tests for the unified question-type registry."""

import pytest

from lexi_ai.contracts.questions import QuestionTypeInfo, RenderKind
from lexi_ai.questions import base
from lexi_ai.questions.types import _shared as shared
from lexi_ai.read_models import Entry, SenseView


class _Assessment:
    info = QuestionTypeInfo(
        type_id="definition_mcq",
        render_kind=RenderKind.SINGLE_CHOICE,
        interaction="assessment",
        difficulty_levels=frozenset({1}),
    )

    async def prepare(self, ctx, demands):
        return base.PrepareReport({})

    async def retrieve(self, ctx, query):
        return None

    async def grade(self, ctx, persisted, submission):
        raise NotImplementedError


def _info(**overrides) -> QuestionTypeInfo:
    values = {
        "type_id": "definition_mcq",
        "render_kind": RenderKind.SINGLE_CHOICE,
        "interaction": "assessment",
        "difficulty_levels": frozenset({1}),
    }
    values.update(overrides)
    return QuestionTypeInfo(**values)


def _plugin(info, *, grade=True):
    attrs = {
        "info": info,
        "prepare": lambda self, ctx, demands: None,
        "retrieve": lambda self, ctx, query: None,
    }
    if grade:
        attrs["grade"] = lambda self, ctx, persisted, submission: None
    return type("FakeType", (), attrs)


class _FakeEntryPoint:
    def __init__(self, name, factory):
        self.name = name
        self._factory = factory

    def load(self):
        return self._factory


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
    ("info", "message"),
    [
        (_info(type_id="unknown"), "unknown question type"),
        (_info(interaction="unknown"), "unknown interaction mode"),
        (_info(difficulty_levels=frozenset()), "difficulty_levels"),
        (_info(difficulty_levels=frozenset({5})), "difficulty_levels"),
        (
            _info(
                type_id="flashcard",
                render_kind=RenderKind.SINGLE_CHOICE,
                difficulty_levels=frozenset({0}),
                interaction="exposure",
            ),
            "level 0",
        ),
        (
            _info(
                type_id="flashcard",
                render_kind=RenderKind.FLASHCARD,
                difficulty_levels=frozenset({0, 1}),
                interaction="exposure",
            ),
            "level 0",
        ),
    ],
)
def test_register_rejects_invalid_info(info, message):
    with pytest.raises(ValueError, match=message):
        base.register(_plugin(info))


def test_register_rejects_assessment_without_grade():
    with pytest.raises(ValueError, match="grade"):
        base.register(_plugin(_info(), grade=False))


def test_register_rejects_exposure_with_grade():
    info = _info(
        type_id="flashcard",
        render_kind=RenderKind.FLASHCARD,
        difficulty_levels=frozenset({0}),
        interaction="exposure",
    )
    with pytest.raises(ValueError, match="must not define grade"):
        base.register(_plugin(info, grade=True))


def test_load_entry_point_types_gates_on_allowlist_and_skips_registered(monkeypatch):
    calls = []
    alpha, beta, gamma = object(), object(), object()
    eps = [
        _FakeEntryPoint("alpha", alpha),
        _FakeEntryPoint("beta", beta),
        _FakeEntryPoint("gamma", gamma),
    ]
    monkeypatch.setattr(base, "entry_points", lambda group: eps)
    monkeypatch.setattr(base, "register", lambda make_plugin: calls.append(make_plugin))
    # A type already in the registry is never re-registered even if allowlisted.
    base.REGISTRY["beta"] = object()

    base.load_entry_point_types({"alpha", "beta"})

    # gamma is not allowlisted; beta is already registered -> only alpha loads.
    assert calls == [alpha]


def test_load_entry_point_types_without_allowlist_is_noop(monkeypatch):
    def _boom(group):
        raise AssertionError("entry points must not be scanned without an allowlist")

    monkeypatch.setattr(base, "entry_points", _boom)
    base.load_entry_point_types(None)
    base.load_entry_point_types(set())


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


def test_mcq_builder_stamps_persisted_carrier():
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
    assert question.question_id is None  # a draft until the store stamps an id
    assert question.type_id == "definition_mcq"
    assert question.render_kind is RenderKind.SINGLE_CHOICE
    assert question.difficulty_level == 1
    assert question.interaction == "assessment"
    assert not hasattr(question, "render_format")
    assert not hasattr(question, "answer_kind")


def test_exposure_builder_stamps_flashcard_payload():
    entry, sense = _entry()
    question = shared._exposure_question(entry, sense)
    assert question.type_id == "flashcard"
    assert question.render_kind is RenderKind.FLASHCARD
    assert question.difficulty_level == 0
    assert question.interaction == "exposure"
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
