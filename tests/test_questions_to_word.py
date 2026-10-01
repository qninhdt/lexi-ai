import pytest
from sqlalchemy import select
from test_prompting import bound_content, prompt_context

from lexi_ai.db.session import Database
from lexi_ai.errors import InvalidOutputError, InvalidResourceError
from lexi_ai.inference.config import DecisionConfig
from lexi_ai.questions.generate import generate_questions
from lexi_ai.questions.grade import grade_answer
from lexi_ai.questions.storage import get, list_for_sense, retrieve
from lexi_ai.schema import Base, Definition, Example, Sense, Theme, Word


class LLM:
    def __init__(self, invalid_last=False):
        self.calls = 0
        self.invalid_last = invalid_last

    async def complete(self, instruction, data, schema):
        self.calls += 1
        context = prompt_context(data)
        kind = context["question_type"]
        content = bound_content(context)
        if content is None:
            content = (
                "The plane _ at dawn."
                if kind == "cloze_to_word"
                else "A situation calling for the target Word."
            )
        correct = "bank" if kind == "definition_to_word" else "took off"
        questions = [
            {
                "content": content,
                **(
                    {"correct_explanation": "It fits."}
                    if schema.__name__ == "AnchoredQuestionBatch"
                    else {"correct": {"content": correct, "explanation": "It fits."}}
                ),
                "distractors": [
                    {"content": f"wrong{i}", "explanation": "Not here."}
                    for i in range(context["distractors_per_question"])
                ],
            }
            for _ in range(context["count"])
        ]
        if self.invalid_last:
            questions[-1]["distractors"] = []
        return schema.model_validate({"questions": questions})


@pytest.fixture
async def setup(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}")
    await db.create_schema(Base.metadata)
    async with db.transaction() as session:
        word = Word(lemma="bank", match_key="bank", entry_type="word", generation_state="done")
        session.add(word)
        await session.flush()
        sense = Sense(word_id=word.id, pos="noun", tier="core")
        session.add(sense)
        await session.flush()
        session.add_all(
            [
                Definition(sense_id=sense.id, content="A place for money"),
                Example(sense_id=sense.id, content='The <t inf="base">bank</t> opens.'),
            ]
        )
    yield db, sense.id
    await db.close()


async def test_batch_append_reuses_artifact_across_formats(setup):
    db, sense_id = setup
    llm = LLM()
    first = await generate_questions(db, llm, sense_id, "definition_to_word", 2, distractor_count=4)
    second = await generate_questions(
        db, llm, sense_id, "definition_to_word", 2, distractor_count=4
    )
    assert len({q.id for q in [*first, *second]}) == 4
    assert llm.calls == 2
    assert len(await list_for_sense(db, sense_id)) == 4
    assert first[0].supports("single_choice") and first[0].supports("single_word")


async def test_last_invalid_question_rolls_back(setup):
    db, sense_id = setup
    with pytest.raises(InvalidOutputError):
        await generate_questions(
            db, LLM(invalid_last=True), sense_id, "definition_to_word", 2, distractor_count=3
        )
    assert await list_for_sense(db, sense_id) == []


async def test_cloze_saves_inflected_answer_not_headword(setup):
    db, sense_id = setup
    question = (
        await generate_questions(db, LLM(), sense_id, "cloze_to_word", 1, distractor_count=3)
    )[0]
    assert question.content.count("_") == 1
    assert question.correct.content == "took off"


async def test_definition_question_uses_current_single_definition(setup):
    db, sense_id = setup
    async with db.transaction() as session:
        definition = await session.scalar(select(Definition).where(Definition.sense_id == sense_id))
        definition.content = "An edge beside a river"
    question = (
        await generate_questions(db, LLM(), sense_id, "definition_to_word", 1, distractor_count=3)
    )[0]
    assert question.content == "An edge beside a river"
    assert (await list_for_sense(db, sense_id))[0].content == question.content


async def test_missing_themed_word_never_generates_neutral_question(setup):
    db, sense_id = setup
    async with db.transaction() as session:
        theme = Theme(key="pirate", name="Pirate", voice="Captain", diction="nautical")
        session.add(theme)
        await session.flush()
    llm = LLM()
    with pytest.raises(InvalidResourceError, match="namespace"):
        await generate_questions(
            db, llm, sense_id, "definition_to_word", 1, distractor_count=3, theme_id=theme.id
        )
    assert llm.calls == 0
    assert await list_for_sense(db, sense_id, theme_id=theme.id) == []
    assert await list_for_sense(db, sense_id) == []


@pytest.mark.parametrize("kind", ["definition_to_word", "word_to_definition", "context_to_word"])
async def test_anchored_correct_answer_is_not_in_model_schema(setup, kind):
    db, sense_id = setup

    class AnchoredLLM(LLM):
        async def complete(self, instruction, data, schema):
            item_schema = schema.model_json_schema()["$defs"]["AnchoredQuestion"]
            assert "correct" not in item_schema["properties"]
            assert "correct" not in item_schema["required"]
            assert "Do NOT generate a `correct` object" not in instruction
            assert "Do not generate or rewrite" not in instruction
            assert "Return only `content`" not in instruction
            context = prompt_context(data)
            assert context["definition" if kind == "word_to_definition" else "word"] == (
                "A place for money" if kind == "word_to_definition" else "bank"
            )
            return await super().complete(instruction, data, schema)

    question = (await generate_questions(db, AnchoredLLM(), sense_id, kind, 1, distractor_count=3))[
        0
    ]
    assert question.correct.content == (
        "A place for money" if kind == "word_to_definition" else "bank"
    )
    assert question.correct.explanation == "It fits."
    assert await get(db, question.id) == question


def dialogue_output(placement):
    text = (
        'The <t inf="base">bank</t> is closed.'
        if placement == "dialogue"
        else "Where should I deposit my savings?"
    )
    answer = 'The <t inf="base">bank</t> can keep them safe.'
    return {
        "questions": [
            {
                "content": [
                    {"speaker": "Maya", "text": text},
                    {"speaker": "Leo", "text": None},
                ],
                "correct": {"content": answer, "explanation": "Fits."},
                "distractors": [
                    {
                        "content": (
                            f'The <t inf="base">bank</t> reply {i}.'
                            if placement == "options"
                            else f"Wrong reply {i}."
                        ),
                        "explanation": "Does not fit.",
                    }
                    for i in range(3)
                ],
            }
        ]
    }


@pytest.mark.parametrize("placement", [None, "dialogue", "options"])
async def test_dialogue_placement_is_saved_retrieved_and_graded(setup, placement):
    db, sense_id = setup
    expected = placement or "dialogue"

    class DialogueLLM:
        async def complete(self, instruction, data, schema):
            assert prompt_context(data)["target_placement"] == expected
            assert ("Target in Options" in instruction) == (expected == "options")
            return dialogue_output(expected)

    question = (
        await generate_questions(
            db,
            DialogueLLM(),
            sense_id,
            "dialogue_completion",
            1,
            distractor_count=3,
            target_placement=placement,
        )
    )[0]
    assert question.target_placement == expected
    assert await get(db, question.id) == question
    assert await retrieve(db, sense_id) == question
    assert (
        await grade_answer(
            db,
            None,
            question.id,
            "single_choice",
            question.correct.id,
            config=DecisionConfig(0.8),
        )
    ).task_fit


@pytest.mark.parametrize(
    "case",
    ["visible_target", "missing_tag", "wrong_target", "malformed_tag", "empty_turn"],
)
async def test_invalid_dialogue_options_do_not_publish_partial_batch(setup, case):
    db, sense_id = setup

    class InvalidDialogueLLM:
        async def complete(self, instruction, data, schema):
            payload = dialogue_output("options")
            invalid = dialogue_output("options")["questions"][0]
            if case == "visible_target":
                invalid["content"][0]["text"] = "The bank is closed."
            elif case == "missing_tag":
                invalid["distractors"][0]["content"] = "The bank is closed."
            elif case == "wrong_target":
                invalid["correct"]["content"] = 'The <t inf="base">vault</t> is closed.'
            elif case == "malformed_tag":
                invalid["correct"]["content"] = 'The <t inf="base">bank is closed.'
            else:
                invalid["content"][0]["text"] = " "
            payload["questions"].append(invalid)
            return payload

    with pytest.raises(InvalidOutputError):
        await generate_questions(
            db,
            InvalidDialogueLLM(),
            sense_id,
            "dialogue_completion",
            2,
            distractor_count=3,
            target_placement="options",
        )
    assert await list_for_sense(db, sense_id) == []


@pytest.mark.parametrize(
    "kind,placement", [("dialogue_completion", "bad"), ("context_to_word", "options")]
)
async def test_invalid_placement_is_rejected_before_model_call(setup, kind, placement):
    db, sense_id = setup
    llm = LLM()
    with pytest.raises(InvalidResourceError, match="target placement"):
        await generate_questions(
            db, llm, sense_id, kind, 1, distractor_count=3, target_placement=placement
        )
    assert llm.calls == 0
