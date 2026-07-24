"""Acceptance tests for the five MVP question types (unified plugin model)."""

from dataclasses import replace

import pytest

from lexi_ai.contracts.questions import (
    AnswerSubmission,
    ChoiceResponse,
    ChoiceReveal,
    RenderKind,
    SpanReveal,
    TextResponse,
)
from lexi_ai.domain.questions import PersistedQuestion
from lexi_ai.questions.base import (
    REGISTRY,
    PrepareReport,
    QuestionContext,
    QuestionDemand,
    QuestionQuery,
)
from lexi_ai.questions.schemas import GeneratedMCQ, Judgment
from lexi_ai.questions.types import (
    Cloze,
    ContextualMCQ,
    DefinitionMCQ,
    Flashcard,
    UseInSentence,
)
from lexi_ai.read_models import Entry, SenseView


class FakeDistractors:
    def __init__(self, pool=("terse", "clumsy", "dull")):
        self.pool = list(pool)

    async def for_word(self, entry, *, k, pos=None):
        return self.pool[:k]


class FakeLLM:
    def __init__(self, value):
        self.value = value

    async def parse(self, messages, schema):
        return self.value


class FakeStore:
    """Content-deduplicating store with exact-level retrieval (PersistedQuestion)."""

    def __init__(self):
        self.questions: list[PersistedQuestion] = []

    @staticmethod
    def _identity(question: PersistedQuestion):
        return (
            question.sense_id,
            question.type_id,
            question.difficulty_level,
            repr(sorted(question.payload.items())),
        )

    async def insert(self, draft: PersistedQuestion) -> PersistedQuestion:
        for existing in self.questions:
            if self._identity(existing) == self._identity(draft):
                return existing
        stored = replace(draft, question_id=len(self.questions) + 1)
        self.questions.append(stored)
        return stored

    async def retrieve_one(
        self, sense_id, difficulty_level, type_id, excluded_ids
    ) -> PersistedQuestion | None:
        return next(
            (
                question
                for question in self.questions
                if question.sense_id == sense_id
                and question.difficulty_level == difficulty_level
                and question.type_id == type_id
                and question.question_id not in excluded_ids
            ),
            None,
        )

    async def list_for_word(self, word_id, type_id=None):
        return [
            question
            for question in self.questions
            if question.word_id == word_id
            and (type_id is None or question.type_id == type_id)
        ]

    async def get(self, question_id):
        return next(
            (question for question in self.questions if question.question_id == question_id),
            None,
        )

    async def delete(self, question_id):
        before = len(self.questions)
        self.questions = [
            question for question in self.questions if question.question_id != question_id
        ]
        return len(self.questions) != before


class FakeSenseLoader:
    def __init__(self, entry: Entry | None):
        self.entry = entry
        self.calls: list[int] = []

    async def load_entry(self, sense_id: int) -> Entry | None:
        self.calls.append(sense_id)
        return self.entry


def _entry() -> Entry:
    return Entry(
        display="eloquent",
        norm="eloquent",
        entry_type="word",
        pos="adjective",
        status="done",
        word_id=3,
        senses=[
            SenseView(
                definition="fluent and persuasive in speech",
                tier="core",
                pos="adjective",
                cefr_level="C1",
                examples=["She gave an eloquent speech."],
                ipa_uk="ˈel.ə.kwənt",
                forms=[],
                sense_id=7,
            )
        ],
    )


def _ctx(*, llm=None, judge=None, store=None, sense_loader=None, distractors=None):
    return QuestionContext(
        entry=_entry(),
        distractors=distractors or FakeDistractors(),
        llm=llm,
        judge=judge,
        store=store,
        sense_loader=sense_loader,
    )


def _demand(level: int, expected_count: int = 1) -> list[QuestionDemand]:
    return [QuestionDemand(sense_id=7, difficulty_level=level, expected_count=expected_count)]


def _submit(question: PersistedQuestion, response) -> AnswerSubmission:
    return AnswerSubmission(question_id=str(question.question_id), response=response)


EXPECTED_INFO = {
    "flashcard": (RenderKind.FLASHCARD, frozenset({0}), "exposure"),
    "definition_mcq": (RenderKind.SINGLE_CHOICE, frozenset({1}), "assessment"),
    "contextual_mcq": (RenderKind.SINGLE_CHOICE, frozenset({1, 2}), "assessment"),
    "cloze": (RenderKind.TEXT_SPAN, frozenset({2, 3}), "assessment"),
    "use_in_sentence": (RenderKind.FREE_TEXT, frozenset({3, 4}), "assessment"),
}


def test_registry_contains_exactly_five_mvp_types():
    assert set(REGISTRY) == set(EXPECTED_INFO)
    for type_id, expected in EXPECTED_INFO.items():
        info = REGISTRY[type_id].info
        assert (info.render_kind, info.difficulty_levels, info.interaction) == expected


@pytest.mark.parametrize(
    "descoped",
    ["matching", "listening", "spelling", "pronunciation_mcq", "collocation_fill"],
)
def test_descoped_types_are_not_registered(descoped):
    assert descoped not in REGISTRY


@pytest.mark.parametrize(
    ("plugin", "level"),
    [(DefinitionMCQ(), 1), (Cloze(), 2), (Cloze(), 3), (UseInSentence(), 3), (UseInSentence(), 4)],
)
async def test_rule_type_prepare_persists_once_and_reports_supply(plugin, level):
    store = FakeStore()
    ctx = _ctx(store=store)

    first = await plugin.prepare(ctx, _demand(level, expected_count=5))
    second = await plugin.prepare(ctx, _demand(level, expected_count=5))

    assert first == PrepareReport({(7, level): 1})
    assert second == PrepareReport({(7, level): 1})
    assert len(store.questions) == 1
    assert store.questions[0].difficulty_level == level


async def test_prepare_ignores_unsupported_levels_and_non_positive_demand():
    store = FakeStore()
    report = await DefinitionMCQ().prepare(
        _ctx(store=store),
        [
            QuestionDemand(7, 2, 1),
            QuestionDemand(7, 1, 0),
            QuestionDemand(7, 1, -1),
        ],
    )
    assert report == PrepareReport({})
    assert store.questions == []


async def test_contextual_l1_is_direct_and_does_not_require_llm():
    store = FakeStore()
    report = await ContextualMCQ().prepare(_ctx(store=store), _demand(1))
    assert report.produced == {(7, 1): 1}
    assert store.questions[0].payload["stem"].startswith("Which word means:")


async def test_contextual_l2_uses_contextual_llm_stem():
    store = FakeStore()
    generated = GeneratedMCQ(
        stem="His ____ speech moved the audience.",
        correct="eloquent",
        distractors=["terse", "dull"],
    )
    report = await ContextualMCQ().prepare(
        _ctx(store=store, llm=FakeLLM(generated)), _demand(2)
    )
    assert report.produced == {(7, 2): 1}
    assert store.questions[0].payload["stem"] == generated.stem


async def test_contextual_l2_without_llm_reports_zero_supply():
    store = FakeStore()
    report = await ContextualMCQ().prepare(_ctx(store=store), _demand(2))
    assert report.produced == {(7, 2): 0}
    assert store.questions == []


async def test_retrieve_is_store_only_exact_level_and_honors_exclusion():
    store = FakeStore()
    plugin = DefinitionMCQ()
    ctx = _ctx(store=store)
    await plugin.prepare(ctx, _demand(1))

    found = await plugin.retrieve(ctx, QuestionQuery(7, 1))
    assert found is not None and found.difficulty_level == 1
    assert isinstance(found, PersistedQuestion)
    assert await plugin.retrieve(ctx, QuestionQuery(7, 2)) is None
    assert await plugin.retrieve(
        ctx, QuestionQuery(7, 1, frozenset({found.question_id}))
    ) is None
    assert await plugin.retrieve(ctx, QuestionQuery(999, 1)) is None


@pytest.mark.parametrize(("level", "has_bank"), [(2, True), (3, False)])
async def test_cloze_level_controls_word_bank(level, has_bank):
    store = FakeStore()
    await Cloze().prepare(_ctx(store=store), _demand(level))
    bank = store.questions[0].payload["word_bank"]
    assert bool(bank) is has_bank
    if has_bank:
        assert "eloquent" in bank


async def test_definition_mcq_grading_returns_typed_choice_reveal():
    store = FakeStore()
    ctx = _ctx(store=store)
    await DefinitionMCQ().prepare(ctx, _demand(1))
    question = store.questions[0]
    correct_index = question.payload["correct_index"]

    correct = await DefinitionMCQ().grade(
        ctx, question, _submit(question, ChoiceResponse(selected_index=correct_index))
    )
    incorrect = await DefinitionMCQ().grade(
        ctx, question, _submit(question, TextResponse(text="wrong"))
    )

    assert (correct.status, correct.correct, correct.score) == ("graded", True, 1.0)
    assert (incorrect.status, incorrect.correct, incorrect.score) == ("graded", False, 0.0)
    # The correct answer is disclosed ONLY through the typed reveal.
    assert isinstance(correct.reveal, ChoiceReveal)
    assert correct.reveal.correct_index == correct_index
    assert correct.reveal.correct_option == question.payload["options"][correct_index]


async def test_cloze_grading_folds_case_and_reveals_span():
    store = FakeStore()
    ctx = _ctx(store=store)
    await Cloze().prepare(ctx, _demand(3))
    question = store.questions[0]

    correct = await Cloze().grade(ctx, question, _submit(question, TextResponse(text="ELOQUENT")))
    incorrect = await Cloze().grade(ctx, question, _submit(question, TextResponse(text="wrong")))

    assert (correct.status, correct.correct, correct.score) == ("graded", True, 1.0)
    assert (incorrect.status, incorrect.correct, incorrect.score) == ("graded", False, 0.0)
    assert isinstance(correct.reveal, SpanReveal)
    assert correct.reveal.correct_answer == "eloquent"


async def test_use_in_sentence_levels_have_distinct_constraints():
    store = FakeStore()
    plugin = UseInSentence()
    ctx = _ctx(store=store)
    await plugin.prepare(ctx, [*_demand(3), *_demand(4)])
    by_level = {question.difficulty_level: question for question in store.questions}

    assert "at least six words" in by_level[3].payload["prompt"].lower()
    assert "at least 6 words" in by_level[3].payload["rubric"].lower()
    assert "at least" not in by_level[4].payload["prompt"].lower()
    assert "at least" not in by_level[4].payload["rubric"].lower()


async def test_use_in_sentence_grading_is_pending_without_judge():
    store = FakeStore()
    plugin = UseInSentence()
    ctx = _ctx(store=store)
    await plugin.prepare(ctx, _demand(4))
    question = store.questions[0]
    evaluation = await plugin.grade(
        ctx, question, _submit(question, TextResponse(text="An eloquent speaker inspired us."))
    )
    assert evaluation.status == "pending"
    assert evaluation.correct is None and evaluation.score is None
    # The provider-free reader cannot grade free text, so nothing is revealed yet.
    assert evaluation.reveal is None


async def test_use_in_sentence_grading_is_graded_with_judge():
    store = FakeStore()
    plugin = UseInSentence()
    judge = FakeLLM(Judgment(correct=True, score=0.75, feedback="Good."))
    ctx = _ctx(store=store, judge=judge)
    await plugin.prepare(ctx, _demand(4))
    question = store.questions[0]
    evaluation = await plugin.grade(
        ctx, question, _submit(question, TextResponse(text="An eloquent speaker inspired us."))
    )
    assert (evaluation.status, evaluation.correct, evaluation.score, evaluation.feedback) == (
        "graded",
        True,
        0.75,
        "Good.",
    )
    # The rubric reveal carries the judge's feedback, filled at grade time.
    assert evaluation.reveal is not None
    assert evaluation.reveal.feedback == "Good."


async def test_flashcard_uses_narrow_loader_without_prepare_or_store():
    entry = _entry()
    loader = FakeSenseLoader(entry)
    ctx = _ctx(store=None, sense_loader=loader)

    question = await Flashcard().retrieve(ctx, QuestionQuery(7, 0))

    assert loader.calls == [7]
    assert question.question_id is None
    assert question.type_id == "flashcard"
    assert question.interaction == "exposure"
    assert question.payload["word"] == "eloquent"
    assert not hasattr(Flashcard(), "grade")


async def test_flashcard_rejects_missing_capability_or_invalid_sense():
    with pytest.raises(RuntimeError, match="sense_loader"):
        await Flashcard().retrieve(_ctx(), QuestionQuery(7, 0))
    with pytest.raises(LookupError, match="sense 7"):
        await Flashcard().retrieve(
            _ctx(sense_loader=FakeSenseLoader(None)), QuestionQuery(7, 0)
        )


class RaisingPrepareType:
    info = REGISTRY["use_in_sentence"].info

    async def prepare(self, ctx, demands):
        raise RuntimeError("provider unavailable")

    async def retrieve(self, ctx, query):
        return None

    async def grade(self, ctx, persisted, submission):
        raise NotImplementedError


async def test_engine_dispatches_new_contract_best_effort(monkeypatch):
    from lexi_ai.questions.engine import QuestionEngine

    store = FakeStore()
    engine = QuestionEngine(
        store,
        FakeDistractors(),
        sense_loader=FakeSenseLoader(_entry()),
    )
    monkeypatch.setitem(REGISTRY, "use_in_sentence", RaisingPrepareType())

    infos = engine.question_types()
    report = await engine.prepare(
        _entry(),
        [QuestionDemand(7, 1, 1), QuestionDemand(7, 4, 1)],
    )

    assert len(infos) == 5
    assert {info.type_id for info in infos} == set(EXPECTED_INFO)
    assert report.produced[(7, 1)] == 2
    assert report.produced[(7, 4)] == 0


async def test_engine_prepare_foreign_sense_does_not_use_core_sense():
    from lexi_ai.questions.engine import QuestionEngine

    store = FakeStore()
    engine = QuestionEngine(store, FakeDistractors())

    report = await engine.prepare(_entry(), [QuestionDemand(999, 1, 1)])

    assert report.produced == {(999, 1): 0}
    assert store.questions == []


async def test_engine_retrieve_presents_answer_free_and_unknown_type_is_typed():
    from lexi_ai.contracts.questions import PresentedQuestion, SingleChoice
    from lexi_ai.questions.engine import QuestionEngine, UnknownQuestionType

    store = FakeStore()
    engine = QuestionEngine(store, FakeDistractors())
    await DefinitionMCQ().prepare(_ctx(store=store), _demand(1))
    stored = store.questions[0]

    presented = await engine.retrieve(7, 1, frozenset(), "definition_mcq")
    assert isinstance(presented, PresentedQuestion)
    assert presented.question_id == str(stored.question_id)
    assert isinstance(presented.render, SingleChoice)
    # The answer index is NOT reachable on the presentation.
    assert not hasattr(presented.render, "correct_index")

    assert await engine.retrieve(7, 2, frozenset(), "definition_mcq") is None
    assert await engine.retrieve(
        7, 1, frozenset({stored.question_id}), "definition_mcq"
    ) is None
    with pytest.raises(UnknownQuestionType) as exc:
        await engine.retrieve(7, 1, frozenset(), "missing")
    assert exc.value.type_id == "missing"


async def test_engine_retrieve_exposure_uses_sense_loader_and_evaluate_rejects_it():
    from lexi_ai.questions.engine import NotAssessable, QuestionEngine

    loader = FakeSenseLoader(_entry())
    engine = QuestionEngine(FakeStore(), FakeDistractors(), sense_loader=loader)

    presented = await engine.retrieve_exposure(7)

    assert loader.calls == [7]
    assert presented.interaction == "exposure"
    assert presented.question_id == "exposure:7"
    with pytest.raises(NotAssessable):
        await engine.retrieve(7, 0, frozenset(), "flashcard")
    assert loader.calls == [7]


async def test_engine_evaluate_dispatches_assessment_and_rejects_exposure():
    from lexi_ai.questions.engine import NotAssessable, QuestionEngine

    store = FakeStore()
    engine = QuestionEngine(store, FakeDistractors())
    await DefinitionMCQ().prepare(_ctx(store=store), _demand(1))
    question = store.questions[0]

    submission = _submit(
        question, ChoiceResponse(selected_index=question.payload["correct_index"])
    )
    result = await engine.evaluate(question, submission)
    assert (result.status, result.correct, result.score) == ("graded", True, 1.0)

    exposure = PersistedQuestion(
        question_id=None,
        word_id=3,
        sense_id=7,
        type_id="flashcard",
        render_kind=RenderKind.FLASHCARD,
        difficulty_level=0,
        interaction="exposure",
        payload={"word": "eloquent", "definition": "x"},
    )
    with pytest.raises(NotAssessable) as exc:
        await engine.evaluate(exposure, _submit_none())
    assert exc.value.question_id is None


def _submit_none() -> AnswerSubmission:
    return AnswerSubmission(question_id="exposure:7", response=TextResponse(text=""))
