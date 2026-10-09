"""Minimal answer synthesis, followed by blind labeling with Lexi's grading rubrics."""

import asyncio
from collections import defaultdict
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel
from typesafe_sdk import Noul

from lexi_ai import DecisionMode
from lexi_ai.config import MAX_QUERY_LENGTH
from lexi_ai.inference.prompting import render_decision, render_prompt
from lexi_ai.inference.usage import merge_usage
from lexi_ai.text import strip_markup
from lexi_ai.vocab import EntryType, QuestionType
from lexi_ai.words.search import search
from lexi_ai.words.storage import meaning_inventory

from .run import PROMPTS, prepare_case

QUESTION_TYPES = (
    QuestionType.DEFINITION_TO_WORD,
    QuestionType.CONTEXT_TO_WORD,
    QuestionType.CLOZE_TO_WORD,
    QuestionType.WORD_TO_DEFINITION,
    QuestionType.WORD_TO_USAGE,
)
GROUP_WEIGHTS = {"word": 30, "phrasal_verb": 12, "idiom": 9, "phrase": 6, "expression": 3}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Entry(Strict):
    target: str = Field(min_length=1)
    source_id: int = Field(gt=0)
    group: Literal["word", "phrasal_verb", "idiom", "phrase", "expression"]


class ThemeSpec(Strict):
    key: str = Field(min_length=1)
    name: str = Field(min_length=1)
    concept: str = Field(min_length=1)


class GenerationConfig(Strict):
    entries: list[Entry] = Field(min_length=1)
    themes: list[ThemeSpec] = Field(default_factory=list, max_length=3)
    examples_per_sense: int = Field(default=3, gt=0)
    distractors_per_question: int = Field(default=3, ge=3, le=20)
    questions_per_type: int = Field(default=12, gt=0)
    workers: int = Field(default=4, gt=0)
    timeout: float = Field(default=120, gt=0, le=120)
    max_completion_tokens: int = Field(default=16384, gt=0, le=16384)
    reasoning_effort: str | None = None


def question_plan(config):
    """Sparse balanced allocation, never the Word × Theme × type cross product."""
    groups = defaultdict(list)
    for index, entry in enumerate(config.entries):
        groups[entry.group].append(index)
    weights = {group: GROUP_WEIGHTS[group] for group in groups}
    credit = dict.fromkeys(groups, 0)
    visits = dict.fromkeys(groups, 0)
    entry_visits = defaultdict(int)
    namespaces = [None, None]
    if config.themes:
        namespaces = [None, config.themes[0].key, None, *[theme.key for theme in config.themes[1:]]]
    jobs = []
    for index in range(config.questions_per_type * len(QUESTION_TYPES)):
        for group, weight in weights.items():
            credit[group] += weight
        group = max(credit, key=credit.get)
        credit[group] -= sum(weights.values())
        entry_index = groups[group][visits[group] % len(groups[group])]
        visits[group] += 1
        jobs.append(
            {
                "id": f"q{index + 1:03d}",
                "entry": entry_index,
                "question_type": QUESTION_TYPES[index % len(QUESTION_TYPES)],
                "theme": namespaces[(index + index // len(QUESTION_TYPES)) % len(namespaces)],
                "sense_index": entry_visits[entry_index],
            }
        )
        entry_visits[entry_index] += 1
    return jobs


# Coverage identifiers only. All blueprint instructions live in the Jinja prompt.
SINGLE_WORD = (
    "canonical",
    "typo",
    "wrong_meaning",
    "variant",
    "wrong_with_typo",
    "alternative",
    "whole_sentence",
    "confusion",
    "irrelevant",
    "injection",
)
DEFINITION = (
    "paraphrase",
    "partial",
    "minimal",
    "misconception",
    "inaccurate",
    "synonym",
    "other_sense",
    "no_candidate",
    "surface_error",
)
USAGE = (
    "correct",
    "spelling",
    "meaning",
    "construction",
    "collocation",
    "acceptable",
    "appropriacy",
    "marked",
    "approximate",
    "multiple",
    "absent",
    "synonym_only",
    "outside_error",
)


def answer_types(job, word, sense):
    kind = job["question_type"]
    pool = (
        SINGLE_WORD
        if kind in QUESTION_TYPES[:3]
        else (DEFINITION if kind == QuestionType.WORD_TO_DEFINITION else USAGE)
    )
    pool = list(pool)
    if sense.forms and kind != QuestionType.WORD_TO_DEFINITION:
        pool.append("inflection")
    if word.type == EntryType.PHRASAL_VERB and kind in QUESTION_TYPES[:3]:
        pool.append("particle")
    if len(word.senses) < 2:
        pool = [item for item in pool if item != "other_sense"]
    # Per-type ordinal advances through the pool rather than resetting each Question.
    offset = (int(job["id"][1:]) - 1) // len(QUESTION_TYPES) * 3
    return [pool[(offset + index) % len(pool)] for index in range(3)]


AnswerText = Annotated[str, Field(strict=True, min_length=1)]


class AnswerBatch(Strict):
    """Native structured outputs require an object at the root, not an array."""

    answers: list[AnswerText]


class AnswerList(RootModel[list[AnswerText]]):
    """Prompted-text output can be the JSON array itself."""


def synthesis_context(question, word, kinds):
    context = {"target": word.lemma, "answer_types": kinds}
    kind = question.question_type
    if kind == QuestionType.DEFINITION_TO_WORD:
        context["definition"] = question.content
    elif kind == QuestionType.CONTEXT_TO_WORD:
        context["context"] = question.content
    elif kind == QuestionType.CLOZE_TO_WORD:
        context["target"] = strip_markup(question.correct.content)
        context["sentence"] = question.content
    elif kind == QuestionType.WORD_TO_DEFINITION:
        context["definition"] = question.correct.content
    elif kind == QuestionType.WORD_TO_USAGE:
        tagged, separator, meaning = question.content.partition("</t> — ")
        if not separator or not meaning:
            raise ValueError("Question has no saved Word/meaning anchor")
        context["target"] = strip_markup(tagged + "</t>")
        context["definition"] = meaning
    else:
        raise ValueError("unsupported synthetic Question type")
    return context


async def synthesize(llm, context, *, question_type, structured_outputs=True):
    if not context["answer_types"]:
        raise ValueError("answer_types must not be empty")
    instruction, data = render_prompt(
        "questions/prompts/generate_synthetic_answers.jinja",
        context=context,
        question_type=question_type,
    )
    schema = AnswerBatch if structured_outputs else AnswerList
    response, usage = await llm.complete(instruction, data, schema, with_usage=True)
    response = schema.model_validate(response)
    answers = response.answers if structured_outputs else response.root
    if len(answers) != len(context["answer_types"]) or any(not text.strip() for text in answers):
        raise ValueError("synthetic output must contain exactly K nonblank strings")
    return answers, usage


def inventory(word):
    return {
        "lemma": word.lemma,
        "senses": [
            {"id": sense.id, "pos": sense.pos, "definition": sense.definition.content}
            for sense in word.senses
        ],
    }


def stage_contexts(question, word, sense):
    base = {"answer": "<learner answer>"}
    meanings = inventory(word)
    if question.question_type in QUESTION_TYPES[:3]:
        return {
            "grade_single_word_1": {
                **base,
                "question": question.content,
                "question_type": question.question_type,
            },
            "grade_single_word_2": {
                **base,
                "question": question.content,
                "matched_word": meanings,
            },
        }
    if question.question_type == QuestionType.WORD_TO_DEFINITION:
        return {
            "grade_word_to_definition_1": {**base, "word": meanings},
            "grade_word_to_definition_2": {
                **base,
                "word": {"lemma": word.lemma},
                "selectedSense": next(
                    item for item in meanings["senses"] if item["id"] == sense.id
                ),
            },
        }
    tagged, separator, meaning = question.content.partition("</t> — ")
    if not separator or not meaning:
        raise ValueError("Question has no saved Word/meaning anchor")
    anchor = {"lemma": strip_markup(tagged + "</t>")}
    return {
        "grade_word_to_usage_1": {**base, "word": anchor},
        "grade_word_to_usage_2": {**base, "word": anchor, "meaning": meaning},
    }


async def grade_answers(lexicon, model, job, question, word, sense, answers):
    """Grade actual answers, never the requested types; record each executed stage."""
    contexts = stage_contexts(question, word, sense)
    cases, usage = [], []
    for index, answer in enumerate(answers, 1):
        labels = {}
        for task, original in contexts.items():
            context = {**original, "answer": answer}
            if task == "grade_single_word_2":
                if not labels["task_fit"] or labels["spelling_error"]:
                    continue
                if len(answer.strip()) > MAX_QUERY_LENGTH:
                    continue
                # Match the production gate and selected top-ranked Word, not a
                # candidate invented by the answer generator.
                result = await search(lexicon.db, answer, limit=1)
                if not result.items:
                    continue
                matched = await meaning_inventory(lexicon.db, word_id=result.items[0].word_id)
                if matched is None or not matched["senses"]:
                    continue
                context["matched_word"] = matched
            if task == "grade_word_to_definition_2":
                selected = labels["defined_meaning"]
                if selected == "no_candidate":
                    continue
                candidates = {f"sense_{value['id']}": value for value in inventory(word)["senses"]}
                context["selectedSense"] = candidates[selected]
            if task == "grade_word_to_usage_2" and labels["used"] is not True:
                continue
            state, questions = render_decision(PROMPTS + task + ".json", **context)
            async with asyncio.timeout(model.llm_config.timeout):
                response, reported = await model.decide(
                    state,
                    questions,
                    mode=DecisionMode.LLM_ONLY,
                    with_usage=True,
                )
            usage.extend(reported)
            expected = {
                name: model.config.accepts(response.nouls[name].noul)
                if isinstance(rule, Noul)
                else response.choices[name].choice
                for name, rule in questions.items()
            }
            labels.update(expected)
            case = {
                "id": f"{job['id']}-a{index}-{task}",
                "task": task,
                "input": context,
                "expected": expected,
                "label_source": "llm_grading",
                "label_status": "proposed",
                "question_id": question.id,
                "answer_id": f"{job['id']}-a{index}",
                "theme": job["theme"],
            }
            prepare_case(case)
            cases.append(case)
    return cases, merge_usage(usage)
