"""Canonical public tokens and actual saved artifacts."""

import json
from dataclasses import asdict

import pytest
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import insert

from lexi_ai import ALLOWED_PAIRS, QUESTION_FORMATS, Lexicon, QuestionType, ResponseFormat
from lexi_ai.db.session import Database
from lexi_ai.models import DefinitionGrade, Option, Question, UsageGrade
from lexi_ai.questions import storage
from lexi_ai.questions.generate import generate_questions
from lexi_ai.questions.schemas import AnchoredQuestionBatch
from lexi_ai.schema import Base, Definition, Sense, Word
from lexi_ai.vocab import (
    EntryType,
    GenerationState,
    PartOfSpeech,
    TargetPlacement,
    Tier,
    normalize_pos,
)


@pytest.fixture
async def dictionary(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'dictionary.db'}")
    await db.create_schema(Base.metadata)
    async with db.transaction() as session:
        await session.execute(
            insert(Word).values(
                id=1,
                lemma="bank",
                match_key="bank",
                entry_type=EntryType.WORD,
                generation_state=GenerationState.DONE,
            )
        )
        await session.execute(
            insert(Sense).values(
                id=1,
                word_id=1,
                pos=PartOfSpeech.NOUN,
                tier=Tier.COMMON,
            )
        )
        await session.execute(
            insert(Definition).values(
                id=1,
                sense_id=1,
                content="a financial institution",
            )
        )
    try:
        yield db
    finally:
        await db.close()


def artifact(kind, *, dialogue=False):
    return Question(
        id=0,
        sense_id=1,
        theme_id=None,
        question_type=kind,
        content=[{"speaker": "Alex", "text": "At the bank?"}, {"speaker": "Jamie", "text": None}]
        if dialogue
        else "A saved prompt",
        correct=Option("answer", "bank", "Saved explanation"),
        distractors=[Option("wrong", "river", "Different meaning")],
        correct_alternatives=["the bank"],
        target_placement=TargetPlacement.DIALOGUE if dialogue else None,
    )


def test_vocabulary_is_public_canonical_and_complete():
    assert len(QuestionType) == 7
    assert len(ResponseFormat) == 3
    assert len(ALLOWED_PAIRS) == 12
    assert set(QUESTION_FORMATS) == set(QuestionType)
    for kind in QuestionType:
        assert kind.name == kind.value
        assert all(isinstance(fmt, ResponseFormat) for fmt in QUESTION_FORMATS[kind])
        with pytest.raises(ValueError):
            QuestionType(kind.value.lower())
    for fmt in ResponseFormat:
        with pytest.raises(ValueError):
            ResponseFormat(fmt.value.lower())


def test_external_corpus_pos_is_normalized_only_at_its_boundary():
    assert normalize_pos("noun") is PartOfSpeech.NOUN
    assert normalize_pos("n.") is PartOfSpeech.NOUN
    assert normalize_pos("a") is PartOfSpeech.ADJECTIVE
    assert normalize_pos("s") is PartOfSpeech.ADJECTIVE
    assert normalize_pos("r") is PartOfSpeech.ADVERB
    assert normalize_pos("unknown source label") is None
    with pytest.raises(ValueError):
        PartOfSpeech("noun")


def test_detached_values_validate_and_serialize_enums_without_invented_fields():
    value = artifact(QuestionType.DIALOGUE_COMPLETION, dialogue=True)
    data = json.loads(TypeAdapter(Question).dump_json(value))
    assert data["question_type"] == "DIALOGUE_COMPLETION"
    assert data["target_placement"] == "DIALOGUE"
    assert data["content"][1]["text"] is None
    assert data["correct"]["explanation"] == "Saved explanation"
    with pytest.raises(ValidationError):
        Question(**(asdict(value) | {"question_type": "dialogue_completion"}))
    assert json.loads(
        TypeAdapter(DefinitionGrade).dump_json(DefinitionGrade(None, None, None))
    ) == {
        "sense_id": None,
        "accuracy": None,
        "coverage": None,
    }
    assert asdict(UsageGrade(False, None, None, None, None, None)) == {
        "used": False,
        "meaning": None,
        "form": None,
        "construction": None,
        "collocation": None,
        "appropriacy": None,
    }


@pytest.mark.parametrize("kind", list(QuestionType))
async def test_native_bank_roundtrip_and_supported_pairs(dictionary, kind):
    saved = (await storage.append(dictionary, [artifact(kind)]))[0]
    reread = await storage.get(dictionary, saved.id)
    assert reread == saved
    assert reread.question_type is kind
    for fmt in ResponseFormat:
        assert reread.supports(fmt) == ((kind, fmt) in ALLOWED_PAIRS)
    assert await storage.list_for_senses(dictionary, [1], [kind]) == [saved]
    assert await storage.list_for_sense(dictionary, 1, kind) == [saved]
    assert await storage.retrieve(dictionary, 1, kind) == saved
    with pytest.raises(ValueError):
        await storage.list_for_senses(dictionary, [1], [kind.value.lower()])


class NoInference:
    async def complete(self, *args, **kwargs):
        raise AssertionError("No paid or fabricated inference is allowed")


async def test_public_native_grading_uses_enum_and_saved_option(dictionary):
    saved = (await storage.append(dictionary, [artifact(QuestionType.DEFINITION_TO_WORD)]))[0]
    lexicon = Lexicon(session=None, db_url=str(dictionary.engine.url), llm=NoInference())
    try:
        result = await lexicon.grade_answer(saved.id, ResponseFormat.SINGLE_CHOICE, "answer")
        assert asdict(result) == {"task_fit": True, "spelling_error": False, "sense_id": 1}
        with pytest.raises(ValueError):
            await lexicon.grade_answer(saved.id, "single_choice", "answer")
        word = await lexicon.get_word(1)
        assert word.type is EntryType.WORD
        assert word.senses[0].pos is PartOfSpeech.NOUN
    finally:
        await lexicon.close()


class DeterministicQuestions:
    """Only the external completion is replaced; schemas/storage stay native."""

    def __init__(self, kind):
        self.kind = kind

    async def complete(self, instruction, data, output_schema):
        kind = self.kind
        if kind is QuestionType.DEFINITION_TO_WORD:
            content = "a financial institution"
        elif kind is QuestionType.WORD_TO_DEFINITION:
            content = '<t inf="base">bank</t>'
        elif kind is QuestionType.WORD_TO_USAGE:
            content = '<t inf="base">bank</t> — a financial institution'
        elif kind is QuestionType.DIALOGUE_COMPLETION:
            content = [
                {"speaker": "Alex", "text": 'Are you visiting the <t inf="base">bank</t>?'},
                {"speaker": "Jamie", "text": None},
            ]
        elif kind is QuestionType.MEANING_IN_CONTEXT:
            content = 'I keep my savings in the <t inf="base">bank</t>.'
        elif kind is QuestionType.CLOZE_TO_WORD:
            content = "I keep my savings in the _."
        else:
            content = "Which place holds my savings?"
        item = {
            "content": content,
            "distractors": [
                {"content": answer, "explanation": "Not the requested meaning"}
                for answer in ("river", "tree", "dinner")
            ],
        }
        if output_schema is AnchoredQuestionBatch:
            item["correct_explanation"] = "The fixed answer is the requested meaning"
        else:
            item["correct"] = {"content": "bank", "explanation": "A financial institution"}
        return {"questions": [item]}


@pytest.mark.parametrize("kind", list(QuestionType))
async def test_all_seven_native_generation_tasks_use_enum_dispatch(dictionary, kind):
    generated = await generate_questions(
        dictionary, DeterministicQuestions(kind), 1, kind, 1, distractor_count=3
    )
    assert len(generated) == 1
    question = generated[0]
    assert question.question_type is kind
    assert len(question.distractors) == 3
    assert await storage.get(dictionary, question.id) == question
    if kind is QuestionType.DIALOGUE_COMPLETION:
        assert question.target_placement is TargetPlacement.DIALOGUE
        assert question.content[1]["text"] is None
