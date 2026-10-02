import pytest
from test_prompting import bound_content, prompt_context

from lexi_ai.db.session import Database
from lexi_ai.models import Option, Question
from lexi_ai.questions.generate import generate_questions
from lexi_ai.questions.storage import list_for_sense
from lexi_ai.schema import Base, Definition, Example, Sense, Theme, Word
from lexi_ai.themes.service import update_theme
from lexi_ai.vocab import ALLOWED_PAIRS, FORMATS, QUESTION_TYPES


class LLM:
    def __init__(self):
        self.calls = 0

    async def complete(self, instruction, data, schema):
        self.calls += 1
        context = prompt_context(data)
        kind = context["question_type"]
        content = bound_content(context)
        if kind == "dialogue_completion":
            content = [
                {"speaker": "Maya", "text": 'The <t inf="base">bank</t> is closed.'},
                {"speaker": "Leo", "text": None},
            ]
        elif kind == "meaning_in_context":
            content = 'The <t inf="base">bank</t> kept my savings safe.'
        elif kind == "context_to_word":
            content = "I went there to deposit money."
        elif kind == "cloze_to_word":
            content = "I visited the _ to deposit money."
        correct = (
            context["definition"]
            if kind == "word_to_definition"
            else (
                "bank" if kind.endswith("_to_word") else 'The <t inf="base">bank</t> opens at nine.'
            )
        )
        return schema.model_validate(
            {
                "questions": [
                    {
                        "content": content,
                        **(
                            {"correct_explanation": "Fits."}
                            if schema.__name__ == "AnchoredQuestionBatch"
                            else {"correct": {"content": correct, "explanation": "Fits."}}
                        ),
                        "distractors": [
                            {"content": f"wrong{i}", "explanation": "Does not fit."}
                            for i in range(context["distractors_per_question"])
                        ],
                    }
                    for _ in range(context["count"])
                ]
            }
        )


def test_exactly_twelve_question_response_pairs():
    expected = (
        {
            (kind, fmt)
            for kind in ("definition_to_word", "context_to_word", "cloze_to_word")
            for fmt in ("single_choice", "single_word")
        }
        | {
            (kind, fmt)
            for kind in ("word_to_definition", "word_to_usage")
            for fmt in ("single_choice", "short_answer")
        }
        | {
            ("dialogue_completion", "single_choice"),
            ("meaning_in_context", "single_choice"),
        }
    )
    assert ALLOWED_PAIRS == expected
    for kind in QUESTION_TYPES:
        for fmt in FORMATS:
            item = Question(0, 1, None, kind, "content", Option("a", "answer", "reason"), [])
            assert item.supports(fmt) == ((kind, fmt) in expected)


@pytest.mark.parametrize("kind", sorted(QUESTION_TYPES))
@pytest.mark.parametrize("themed", [False, True])
async def test_every_type_batches_into_exact_namespace(kind, themed, tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}")
    try:
        await db.create_schema(Base.metadata)
        async with db.transaction() as session:
            word = Word(lemma="bank", match_key="bank", entry_type="word", generation_state="done")
            theme = Theme(key="pirate", name="Pirate", voice="Captain", diction="nautical")
            session.add_all([word, theme])
            await session.flush()
            sense = Sense(word_id=word.id, pos="noun", tier="core")
            session.add(sense)
            await session.flush()
            session.add_all(
                [
                    Definition(sense_id=sense.id, content="A place for money"),
                    Example(sense_id=sense.id, content='The <t inf="base">bank</t> opened.'),
                    Definition(sense_id=sense.id, theme_id=theme.id, content="A place for coins"),
                    Example(
                        sense_id=sense.id,
                        theme_id=theme.id,
                        content='The <t inf="base">bank</t> holds treasure.',
                    ),
                ]
            )
        theme_id = theme.id if themed else None
        if themed:
            await update_theme(db, theme.key, voice="Admiral", diction="modern naval")

        class NamespaceLLM(LLM):
            async def complete(self, instruction, data, schema):
                context = prompt_context(data)
                assert context["theme"] == (
                    {"voice": "Admiral", "diction": "modern naval"} if themed else None
                )
                assert context["definition"] == (
                    "A place for coins" if themed else "A place for money"
                )
                return await super().complete(instruction, data, schema)

        llm = NamespaceLLM()
        created = await generate_questions(
            db, llm, sense.id, kind, 2, distractor_count=3, theme_id=theme_id
        )
        assert len(created) == 2 and len({item.id for item in created}) == 2
        assert all(item.theme_id == theme_id and len(item.distractors) == 3 for item in created)
        assert llm.calls == 1
        assert await list_for_sense(db, sense.id, kind, theme_id) == created
        assert await list_for_sense(db, sense.id, kind, None if themed else theme.id) == []
    finally:
        await db.close()
