"""Seven explicit generation tasks, one structured request and one atomic N-artifact append."""

import uuid

from lexi_ai.errors import InvalidOutputError, InvalidResourceError, MissingProviderError
from lexi_ai.inference.prompting import render_prompt
from lexi_ai.models import Option, Question
from lexi_ai.vocab import QuestionType, TargetPlacement

from .schemas import (
    ANCHORED_QUESTION_TYPES,
    AnchoredQuestionBatch,
    GeneratedOption,
    GeneratedQuestion,
    QuestionBatch,
    validate_batch,
    validate_targets,
)
from .storage import append, generation_context


def _bound_content(question_type: QuestionType, lemma: str, anchor: str) -> str | None:
    if question_type is QuestionType.DEFINITION_TO_WORD:
        return anchor
    if question_type is QuestionType.WORD_TO_DEFINITION:
        return f'<t inf="base">{lemma}</t>'
    if question_type is QuestionType.WORD_TO_USAGE:
        return f'<t inf="base">{lemma}</t> — {anchor}'
    return None


async def generate_questions(
    db,
    llm,
    sense_id: int,
    question_type: QuestionType,
    count: int,
    *,
    distractor_count: int,
    theme_id: int | None = None,
    target_placement: TargetPlacement | None = None,
    theme_key: str | None = None,
) -> list[Question]:
    try:
        question_type = QuestionType(question_type)
        if target_placement is not None:
            target_placement = TargetPlacement(target_placement)
    except ValueError as exc:
        raise InvalidResourceError("invalid Question type or target placement") from exc
    if type(count) is not int or not 1 <= count <= 20:
        raise InvalidResourceError("invalid Question type or count")
    if not 3 <= distractor_count <= 20:
        raise InvalidResourceError("invalid distractor count")
    if question_type is QuestionType.DIALOGUE_COMPLETION:
        target_placement = (
            TargetPlacement.DIALOGUE if target_placement is None else target_placement
        )
    elif target_placement is not None:
        raise InvalidResourceError("target placement applies only to dialogue completion")
    word, sense, theme = await generation_context(db, sense_id, theme_id, theme_key=theme_key)
    theme_id = theme["id"] if theme else None
    if llm is None:
        raise MissingProviderError("Question generation requires a structured LLM")
    anchor = sense.definition.content
    bound = _bound_content(question_type, word.lemma, anchor)
    fixed_answer = (
        (anchor if question_type is QuestionType.WORD_TO_DEFINITION else word.lemma)
        if question_type in ANCHORED_QUESTION_TYPES
        else None
    )
    context = {
        "word": word.lemma,
        "aliases": word.aliases,
        "sense_pos": sense.pos,
        "sense_tier": sense.tier,
        "definition": anchor,
        "examples": [item.content for item in sense.examples],
        "forms": [item.__dict__ for item in sense.forms],
        "patterns": sense.patterns,
        "collocations": sense.collocations,
        "register": sense.register,
        "cefr_level": sense.cefr_level,
        "usage_note": sense.usage_note,
        "theme": ({"voice": theme["voice"], "diction": theme["diction"]} if theme else None),
        "question_type": question_type,
        "count": count,
        "distractors_per_question": distractor_count,
        "target_placement": target_placement,
    }
    instruction, data = render_prompt(
        "questions/prompts/generate_question.jinja",
        question_type=question_type,
        theme=context["theme"],
        target_placement=target_placement,
        context=context,
    )
    schema = AnchoredQuestionBatch if fixed_answer is not None else QuestionBatch
    try:
        output = schema.model_validate(await llm.complete(instruction, data, schema))
        if fixed_answer is not None:
            batch = QuestionBatch(
                questions=[
                    GeneratedQuestion(
                        content=item.content,
                        correct=GeneratedOption(
                            content=fixed_answer, explanation=item.correct_explanation
                        ),
                        distractors=item.distractors,
                    )
                    for item in output.questions
                ]
            )
        else:
            batch = output
        validate_batch(
            batch, question_type, count, distractor_count, target_placement=target_placement
        )
        for item in batch.questions:
            if bound is not None and item.content != bound:
                raise InvalidOutputError("Question changed its bound definition or Word/meaning")
            validate_targets(item, question_type, word, sense, target_placement=target_placement)
    except ValueError as exc:
        raise InvalidOutputError("invalid Question batch") from exc
    artifacts = [
        Question(
            id=0,
            sense_id=sense_id,
            theme_id=theme_id,
            question_type=question_type,
            content=(
                item.content
                if isinstance(item.content, str)
                else [turn.model_dump() for turn in item.content]
            ),
            correct=Option(uuid.uuid4().hex, item.correct.content, item.correct.explanation),
            distractors=[
                Option(uuid.uuid4().hex, distractor.content, distractor.explanation)
                for distractor in item.distractors
            ],
            correct_alternatives=[],
            target_placement=target_placement,
        )
        for item in batch.questions
    ]
    return await append(db, artifacts)
