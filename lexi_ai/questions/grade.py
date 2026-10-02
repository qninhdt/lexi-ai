"""Two-stage diagnostic grading; sense identification never decides task fit."""

from lexi_ai.config import MAX_QUERY_LENGTH, MAX_TEXT_LENGTH
from lexi_ai.errors import (
    InvalidOutputError,
    InvalidResourceError,
    MissingProviderError,
)
from lexi_ai.inference.config import DecisionConfig, DecisionMode
from lexi_ai.inference.prompting import render_decision
from lexi_ai.models import DefinitionGrade, SingleWordGrade, UsageGrade
from lexi_ai.text import answer_key, parse_marked_example
from lexi_ai.words.search import search
from lexi_ai.words.storage import meaning_inventory

from .storage import get as get_question

_PROMPTS = "questions/prompts/decision/"


def _selected_sense(response, name, senses):
    choice = response.choices[name].choice
    if choice == "no_candidate":
        return None
    by_key = {f"sense_{sense['id']}": sense for sense in senses}
    if choice not in by_key:
        raise InvalidOutputError("invalid decision sense")
    return by_key[choice]


async def _single_word(db, model, question, answer, config, options):
    response = await model.decide(
        *render_decision(
            _PROMPTS + "grade_single_word_1.json",
            question=question.content,
            question_type=question.question_type,
            answer=answer,
        ),
        **options,
    )
    task_fit = config.accepts(response.nouls["task_fit"].noul)
    spelling_error = config.accepts(response.nouls["spelling_error"].noul)
    if not task_fit or spelling_error or len(answer.strip()) > MAX_QUERY_LENGTH:
        return SingleWordGrade(task_fit, spelling_error, None)

    result = await search(db, None, answer, limit=1)
    if not result.words:
        return SingleWordGrade(True, False, None)
    matched_word = await meaning_inventory(db, word_id=result.words[0].word_id)
    if matched_word is None or not matched_word["senses"]:
        return SingleWordGrade(True, False, None)
    response = await model.decide(
        *render_decision(
            _PROMPTS + "grade_single_word_2.json",
            question=question.content,
            answer=answer,
            matched_word=matched_word,
        ),
        **options,
    )
    sense = _selected_sense(response, "matched_sense", matched_word["senses"])
    return SingleWordGrade(True, False, sense["id"] if sense else None)


async def _definition(db, model, question, answer, options):
    word = await meaning_inventory(db, owner_of=question.sense_id)
    if word is None or not word["senses"]:
        raise InvalidResourceError("Question's Word has no published meaning inventory")
    response = await model.decide(
        *render_decision(
            _PROMPTS + "grade_word_to_definition_1.json",
            word=word,
            answer=answer,
        ),
        **options,
    )
    sense = _selected_sense(response, "defined_meaning", word["senses"])
    if sense is None:
        return DefinitionGrade(None, None, None)
    response = await model.decide(
        *render_decision(
            _PROMPTS + "grade_word_to_definition_2.json",
            word=word,
            selectedSense=sense,
            answer=answer,
        ),
        **options,
    )
    return DefinitionGrade(
        sense["id"],
        response.choices["accuracy"].choice,
        response.choices["coverage"].choice,
    )


async def _usage(model, question, answer, config, options):
    # Use the saved Word/meaning, not newly edited dictionary wording.
    tagged, separator, meaning = question.content.partition("</t> — ")
    if not separator or not meaning:
        raise InvalidResourceError("Question has no saved Word/meaning anchor")
    lemma, spans = parse_marked_example(tagged + "</t>")
    if len(spans) != 1:
        raise InvalidResourceError("Question has an invalid Word anchor")
    word = {"lemma": lemma}
    response = await model.decide(
        *render_decision(
            _PROMPTS + "grade_word_to_usage_1.json",
            word=word,
            answer=answer,
        ),
        **options,
    )
    if not config.accepts(response.nouls["used"].noul):
        return UsageGrade(False, None, None, None, None, None)
    response = await model.decide(
        *render_decision(
            _PROMPTS + "grade_word_to_usage_2.json",
            word=word,
            meaning=meaning,
            answer=answer,
        ),
        **options,
    )
    return UsageGrade(
        True,
        response.choices["meaning"].choice,
        response.choices["form"].choice,
        config.accepts(response.nouls["construction"].noul),
        response.choices["collocation"].choice,
        response.choices["appropriacy"].choice,
    )


async def grade_answer(
    db,
    decision_model,
    question_id,
    fmt,
    answer,
    *,
    config: DecisionConfig,
    mode: DecisionMode = DecisionMode.LLM_FALLBACK,
):
    mode = DecisionMode(mode)
    options = {"mode": mode}
    if not isinstance(answer, str) or not answer.strip() or len(answer) > MAX_TEXT_LENGTH:
        raise InvalidResourceError("invalid answer")
    question = await get_question(db, question_id)
    if question is None or not question.supports(fmt):
        raise InvalidResourceError("unknown Question or unsupported response format")
    if fmt == "single_choice":
        if answer == question.correct.id:
            return SingleWordGrade(True, False, question.sense_id)
        if any(answer == option.id for option in question.distractors):
            return SingleWordGrade(False, False, None)
        raise InvalidResourceError("unknown saved option ID")
    if fmt == "single_word" and answer_key(answer) == answer_key(question.correct.content):
        return SingleWordGrade(True, False, question.sense_id)
    if decision_model is None:
        raise MissingProviderError("free-text grading requires a decision provider")
    if fmt == "single_word":
        return await _single_word(db, decision_model, question, answer, config, options)
    if question.question_type == "word_to_definition":
        return await _definition(db, decision_model, question, answer, options)
    return await _usage(decision_model, question, answer, config, options)
