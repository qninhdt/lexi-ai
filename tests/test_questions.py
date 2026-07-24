"""Phase 3 acceptance tests for the five MVP question types."""

from dataclasses import replace

import pytest

from lexi_ai.questions.base import (
    REGISTRY,
    PrepareReport,
    QuestionContext,
    QuestionDemand,
    QuestionQuery,
)
from lexi_ai.questions.formats import (
    Cloze,
    ContextualMCQ,
    DefinitionMCQ,
    Flashcard,
    UseInSentence,
)
from lexi_ai.questions.schemas import GeneratedMCQ, Judgment
from lexi_ai.read_models import Entry, Question, SenseView


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
    """Content-deduplicating store with exact-level retrieval."""

    def __init__(self):
        self.questions: list[Question] = []

    async def insert(self, question: Question) -> Question:
        identity = (
            question.sense_id,
            question.type_id,
            question.difficulty_level,
            repr(sorted(question.payload.items())),
        )
        for existing in self.questions:
            existing_identity = (
                existing.sense_id,
                existing.type_id,
                existing.difficulty_level,
                repr(sorted(existing.payload.items())),
            )
            if existing_identity == identity:
                return existing
        stored = replace(question, question_id=len(self.questions) + 1)
        self.questions.append(stored)
        return stored

    async def retrieve_one(
        self, sense_id, difficulty_level, type_id, excluded_ids
    ) -> Question | None:
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


EXPECTED_DESCRIPTORS = {
    "flashcard": ("flashcard", frozenset({0}), "exposure"),
    "definition_mcq": ("single_choice", frozenset({1}), "assessment"),
    "contextual_mcq": ("single_choice", frozenset({1, 2}), "assessment"),
    "cloze": ("text_span", frozenset({2, 3}), "assessment"),
    "use_in_sentence": ("free_text", frozenset({3, 4}), "assessment"),
}


def test_registry_contains_exactly_five_mvp_types():
    assert set(REGISTRY) == set(EXPECTED_DESCRIPTORS)
    for type_id, expected in EXPECTED_DESCRIPTORS.items():
        descriptor = REGISTRY[type_id].descriptor
        assert (
            descriptor.render_format,
            descriptor.supported_levels,
            descriptor.interaction_mode,
        ) == expected


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


@pytest.mark.parametrize(
    ("plugin", "level", "right_answer", "wrong_answer"),
    [
        (DefinitionMCQ(), 1, None, "wrong"),
        (Cloze(), 3, "ELOQUENT", "wrong"),
    ],
)
async def test_rule_evaluation_returns_graded(plugin, level, right_answer, wrong_answer):
    store = FakeStore()
    ctx = _ctx(store=store)
    await plugin.prepare(ctx, _demand(level))
    question = store.questions[0]
    right = question.payload.get("correct_index") if right_answer is None else right_answer

    correct = await plugin.evaluate(ctx, question, right)
    incorrect = await plugin.evaluate(ctx, question, wrong_answer)

    assert (correct.status, correct.verdict, correct.score) == ("graded", True, 1.0)
    assert (incorrect.status, incorrect.verdict, incorrect.score) == ("graded", False, 0.0)


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


async def test_use_in_sentence_evaluation_is_pending_without_judge():
    store = FakeStore()
    plugin = UseInSentence()
    ctx = _ctx(store=store)
    await plugin.prepare(ctx, _demand(4))
    evaluation = await plugin.evaluate(ctx, store.questions[0], "An eloquent speaker inspired us.")
    assert evaluation.status == "pending"
    assert evaluation.verdict is None and evaluation.score is None
    with pytest.raises(RuntimeError, match="pending"):
        _ = evaluation.is_correct


async def test_use_in_sentence_evaluation_is_graded_with_judge():
    store = FakeStore()
    plugin = UseInSentence()
    judge = FakeLLM(Judgment(correct=True, score=0.75, feedback="Good."))
    ctx = _ctx(store=store, judge=judge)
    await plugin.prepare(ctx, _demand(4))
    evaluation = await plugin.evaluate(ctx, store.questions[0], "An eloquent speaker inspired us.")
    assert (evaluation.status, evaluation.verdict, evaluation.score, evaluation.feedback) == (
        "graded",
        True,
        0.75,
        "Good.",
    )


async def test_flashcard_uses_narrow_loader_without_prepare_or_store():
    entry = _entry()
    loader = FakeSenseLoader(entry)
    ctx = _ctx(store=None, sense_loader=loader)

    question = await Flashcard().retrieve(ctx, QuestionQuery(7, 0))

    assert loader.calls == [7]
    assert question.question_id is None
    assert question.type_id == "flashcard"
    assert question.interaction_mode == "exposure"
    assert question.payload["word"] == "eloquent"
    assert not hasattr(Flashcard(), "evaluate")


async def test_flashcard_rejects_missing_capability_or_invalid_sense():
    with pytest.raises(RuntimeError, match="sense_loader"):
        await Flashcard().retrieve(_ctx(), QuestionQuery(7, 0))
    with pytest.raises(LookupError, match="sense 7"):
        await Flashcard().retrieve(
            _ctx(sense_loader=FakeSenseLoader(None)), QuestionQuery(7, 0)
        )



class RaisingPrepareType:
    descriptor = REGISTRY["use_in_sentence"].descriptor

    async def prepare(self, ctx, demands):
        raise RuntimeError("provider unavailable")

    async def retrieve(self, ctx, query):
        return None

    async def evaluate(self, ctx, question, answer):
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

    descriptors = engine.question_types()
    report = await engine.prepare(
        _entry(),
        [QuestionDemand(7, 1, 1), QuestionDemand(7, 4, 1)],
    )

    assert len(descriptors) == 5
    assert {descriptor.type_id for descriptor in descriptors} == set(EXPECTED_DESCRIPTORS)
    assert report.produced[(7, 1)] == 2
    assert report.produced[(7, 4)] == 0


async def test_engine_prepare_foreign_sense_does_not_use_core_sense():
    from lexi_ai.questions.engine import QuestionEngine

    store = FakeStore()
    engine = QuestionEngine(store, FakeDistractors())

    report = await engine.prepare(_entry(), [QuestionDemand(999, 1, 1)])

    assert report.produced == {(999, 1): 0}
    assert store.questions == []


async def test_engine_retrieve_is_exact_store_only_and_unknown_type_is_typed():
    from lexi_ai.questions.engine import QuestionEngine, UnknownQuestionType

    store = FakeStore()
    engine = QuestionEngine(store, FakeDistractors())
    await DefinitionMCQ().prepare(_ctx(store=store), _demand(1))
    stored = store.questions[0]

    assert await engine.retrieve(7, 1, frozenset(), "definition_mcq") == stored
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

    question = await engine.retrieve_exposure(7)

    assert loader.calls == [7]
    assert question.interaction_mode == "exposure"
    with pytest.raises(NotAssessable):
        await engine.retrieve(7, 0, frozenset(), "flashcard")
    assert loader.calls == [7]
    with pytest.raises(NotAssessable) as exc:
        await engine.evaluate(question, None)
    assert exc.value.question_id is None


async def test_engine_evaluate_dispatches_assessment():
    from lexi_ai.questions.engine import QuestionEngine

    store = FakeStore()
    engine = QuestionEngine(store, FakeDistractors())
    await DefinitionMCQ().prepare(_ctx(store=store), _demand(1))
    question = store.questions[0]

    result = await engine.evaluate(question, question.payload["correct_index"])

    assert (result.status, result.verdict, result.score) == ("graded", True, 1.0)
