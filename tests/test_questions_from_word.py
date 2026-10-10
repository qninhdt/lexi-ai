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
        if kind == "DIALOGUE_COMPLETION":
            content = [
                {"speaker": "Maya", "text": "The [bank] is closed."},
                {"speaker": "Leo", "text": None},
            ]
        elif kind == "MEANING_IN_CONTEXT":
            content = "The [bank] kept my savings safe."
        elif kind == "CONTEXT_TO_WORD":
            content = "I went there to deposit money."
        elif kind == "CLOZE_TO_WORD":
            content = "I visited the _ to deposit money."
        correct = (
            context["definition"]
            if kind == "WORD_TO_DEFINITION"
            else ("bank" if kind.endswith("_to_word") else "The [bank] opens at nine.")
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
                            {
                                "content": f"wrong{self.calls}q{item}x{i}",
                                "explanation": "Does not fit.",
                            }
                            for i in range(context["distractors_per_question"])
                        ],
                    }
                    for item in range(context["count"])
                ]
            }
        )


def test_exactly_twelve_question_response_pairs():
    expected = (
        {
            (kind, fmt)
            for kind in ("DEFINITION_TO_WORD", "CONTEXT_TO_WORD", "CLOZE_TO_WORD")
            for fmt in ("SINGLE_CHOICE", "SINGLE_WORD")
        }
        | {
            (kind, fmt)
            for kind in ("WORD_TO_DEFINITION", "WORD_TO_USAGE")
            for fmt in ("SINGLE_CHOICE", "SHORT_ANSWER")
        }
        | {
            ("DIALOGUE_COMPLETION", "SINGLE_CHOICE"),
            ("MEANING_IN_CONTEXT", "SINGLE_CHOICE"),
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
            word = Word(lemma="bank", match_key="bank", entry_type="WORD", generation_state="DONE")
            theme = Theme(key="pirate", name="Pirate", voice="Captain", diction="nautical")
            session.add_all([word, theme])
            await session.flush()
            sense = Sense(word_id=word.id, pos="NOUN", tier="CORE")
            session.add(sense)
            await session.flush()
            session.add_all(
                [
                    Definition(sense_id=sense.id, content="A place for money"),
                    Example(sense_id=sense.id, content="The [bank] opened."),
                    Definition(sense_id=sense.id, theme_id=theme.id, content="A place for coins"),
                    Example(
                        sense_id=sense.id,
                        theme_id=theme.id,
                        content="The [bank] holds treasure.",
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
