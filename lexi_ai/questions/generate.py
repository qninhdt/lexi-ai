"""Seven explicit generation tasks, one structured request and one atomic N-artifact append."""

import uuid

from pydantic import Field, create_model, model_validator

from lexi_ai.errors import InvalidOutputError, InvalidResourceError, MissingProviderError
from lexi_ai.inference.prompting import render_prompt
from lexi_ai.models import Option, Question
from lexi_ai.text import answer_key, strip_markup
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
from .storage import append, generation_context, list_for_sense


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

    def batch_for(output):
        if fixed_answer is None:
            return output
        return QuestionBatch(
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

    base = AnchoredQuestionBatch if fixed_answer is not None else QuestionBatch
    item_schema = base.model_fields["questions"].annotation.__args__[0]

    def check_output(cls, output):
        if not isinstance(output, dict) or not isinstance(output.get("questions"), list):
            return output
        accepted = []
        first_error = None
        for raw in output["questions"]:
            try:
                item = item_schema.model_validate(raw)
                batch = batch_for(base(questions=[item]))
                validate_batch(
                    batch,
                    question_type,
                    count,
                    distractor_count,
                    target_placement=target_placement,
                )
                question = batch.questions[0]
                if bound is not None and answer_key(strip_markup(question.content)) != answer_key(
                    strip_markup(bound)
                ):
                    raise InvalidOutputError(
                        "Question changed its bound definition or Word/meaning"
                    )
                validate_targets(
                    question, question_type, word, sense, target_placement=target_placement
                )
            except (ValueError, InvalidOutputError) as error:
                if first_error is None:
                    first_error = error
                continue
            accepted.append(item)
        if not accepted:
            raise InvalidOutputError("Question batch has no usable questions") from first_error
        return output | {"questions": accepted}

    schema = create_model(
        base.__name__,
        __base__=base,
        __validators__={"check_output": model_validator(mode="before")(classmethod(check_output))},
        questions=(base.model_fields["questions"].annotation, Field(min_length=1)),
    )
    try:
        output = await llm.complete(instruction, data, schema)
        if not isinstance(output, schema):
            output = schema.model_validate(output)
        batch = batch_for(output)
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

    def key(question):
        content = question.content
        if isinstance(content, list):
            content = tuple(
                (
                    turn["speaker"].casefold(),
                    answer_key(strip_markup(turn["text"])) if turn["text"] is not None else None,
                )
                for turn in content
            )
        else:
            content = answer_key(strip_markup(content))
        return (
            content,
            answer_key(strip_markup(question.correct.content)),
            tuple(sorted(answer_key(strip_markup(o.content)) for o in question.distractors)),
        )

    existing = await list_for_sense(db, sense_id, question_type, theme_id)
    seen = {key(question) for question in existing}
    unique = []
    for question in artifacts:
        identity = key(question)
        if identity not in seen:
            unique.append(question)
            seen.add(identity)
    return await append(db, unique)
