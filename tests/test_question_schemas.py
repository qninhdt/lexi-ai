"""Question-domain target validation must preserve complete lexical expressions."""

import pytest

from lexi_ai.errors import InvalidOutputError
from lexi_ai.models import Form, Sense, Word
from lexi_ai.questions.schemas import QuestionBatch, validate_batch, validate_targets


@pytest.mark.parametrize(
    "target,valid",
    [
        ("[brought|p] the issue [up]", True),
        ("[bring] it [up]", True),
        ("[brought|p] the issue up", False),
        ("[up] [bring]", False),
        ("[bring] [out]", False),
    ],
)
def test_dialogue_target_preserves_all_fixed_components_in_order(target, valid):
    sense = Sense(1, 1, "VERB", "CORE", forms=[Form("brought", "PAST")])
    word = Word(1, "bring up", "PHRASAL_VERB", "DONE", senses=[sense])
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
    validate_batch(batch, "DIALOGUE_COMPLETION", 1, 3, target_placement="DIALOGUE")
    if valid:
        validate_targets(
            batch.questions[0], "DIALOGUE_COMPLETION", word, sense, target_placement="DIALOGUE"
        )
    else:
        with pytest.raises(InvalidOutputError, match="complete licensed target"):
            validate_targets(
                batch.questions[0], "DIALOGUE_COMPLETION", word, sense, target_placement="DIALOGUE"
            )
