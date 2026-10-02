"""Question-domain target validation must preserve complete lexical expressions."""

import pytest

from lexi_ai.errors import InvalidOutputError
from lexi_ai.models import Form, Sense, Word
from lexi_ai.questions.schemas import QuestionBatch, validate_batch, validate_targets


@pytest.mark.parametrize(
    "target,valid",
    [
        ('<t inf="past">brought</t> the issue <t inf="base">up</t>', True),
        ('<t inf="base">bring</t> it <t inf="base">up</t>', True),
        ('<t inf="past">brought</t> the issue up', False),
        ('<t inf="base">up</t> <t inf="base">bring</t>', False),
        ('<t inf="base">bring</t> <t inf="base">out</t>', False),
    ],
)
def test_dialogue_target_preserves_all_fixed_components_in_order(target, valid):
    sense = Sense(1, 1, 0, "verb", "core", forms=[Form("brought", "past")])
    word = Word(1, "bring up", "phrasal_verb", "done", senses=[sense])
    batch = QuestionBatch.model_validate(
        {
            "questions": [
                {
                    "content": [
                        {"speaker": "Maya", "text": f"Please {target}."},
                        {"speaker": "Leo", "text": None},
                    ],
                    "correct": {"content": "I will mention it.", "explanation": "Fits."},
                    "distractors": [
                        {"content": f"Wrong reply {i}", "explanation": "Wrong intention."}
                        for i in range(3)
                    ],
                }
            ]
        }
    )
    validate_batch(batch, "dialogue_completion", 1, 3, target_placement="dialogue")
    if valid:
        validate_targets(
            batch.questions[0], "dialogue_completion", word, sense, target_placement="dialogue"
        )
    else:
        with pytest.raises(InvalidOutputError, match="complete licensed target"):
            validate_targets(
                batch.questions[0], "dialogue_completion", word, sense, target_placement="dialogue"
            )
