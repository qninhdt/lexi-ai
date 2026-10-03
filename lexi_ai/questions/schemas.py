"""Question generation schemas and domain validation."""

import re

from pydantic import BaseModel, ConfigDict, Field

from lexi_ai.errors import InvalidOutputError
from lexi_ai.models import Sense, Word
from lexi_ai.text import answer_key, parse_marked_example, strip_markup
from lexi_ai.vocab import QUESTION_TYPES, QuestionType, TargetPlacement


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GeneratedOption(Strict):
    content: str = Field(description="One answer or confusable wrong answer")
    explanation: str = Field(description="Brief English justification for this option")


class DialogueTurn(Strict):
    speaker: str = Field(description="One-word human name, never A/B/C")
    text: str | None = Field(description="Null only for the one missing reply")


class AnchoredQuestion(Strict):
    content: str = Field(description="Only the prompt content; preserve fixed content if supplied")
    correct_explanation: str = Field(description="Explain why the supplied fixed answer fits")
    distractors: list[GeneratedOption] = Field(
        description="Requested K wrong choices, hardest first; exclude the supplied fixed answer"
    )


class AnchoredQuestionBatch(Strict):
    questions: list[AnchoredQuestion]


ANCHORED_QUESTION_TYPES = frozenset(
    {
        QuestionType.DEFINITION_TO_WORD,
        QuestionType.WORD_TO_DEFINITION,
        QuestionType.CONTEXT_TO_WORD,
    }
)


class GeneratedQuestion(Strict):
    content: str | list[DialogueTurn] = Field(
        description="Only the prompt content; no UI instruction"
    )
    correct: GeneratedOption
    distractors: list[GeneratedOption] = Field(
        description="Requested K wrong choices, hardest first; no correct duplicate"
    )


class QuestionBatch(Strict):
    questions: list[GeneratedQuestion]


_INSTRUCTION = re.compile(
    r"^(?:choose|select|fill in|what is|which option|write the correct)\b", re.I
)
_NAME = re.compile(r"[A-Za-z][a-z]+(?:-[A-Za-z][a-z]+)?\Z")


def validate_batch(
    batch: QuestionBatch,
    kind: QuestionType,
    count: int,
    distractor_count: int,
    *,
    target_placement: TargetPlacement | None = None,
) -> None:
    if kind not in QUESTION_TYPES or len(batch.questions) != count:
        raise InvalidOutputError("Question batch has an unexpected type or size")
    for question in batch.questions:
        if len(question.distractors) != distractor_count:
            raise InvalidOutputError("Question distractor bank is incomplete")
        options = [question.correct, *question.distractors]
        if any(not option.content.strip() or not option.explanation.strip() for option in options):
            raise InvalidOutputError("empty Question option or explanation")
        if len({answer_key(strip_markup(o.content)) for o in options}) != len(options):
            raise InvalidOutputError("correct or distractor option repeated")
        content = question.content
        if kind == QuestionType.DIALOGUE_COMPLETION:
            if target_placement not in set(TargetPlacement):
                raise InvalidOutputError("invalid dialogue target placement")
            if (
                not isinstance(content, list)
                or len(content) < 2
                or sum(turn.text is None for turn in content) != 1
                or any(
                    not _NAME.fullmatch(turn.speaker) or turn.speaker in {"A", "B", "C"}
                    for turn in content
                )
                or any(turn.text is not None and not turn.text.strip() for turn in content)
            ):
                raise InvalidOutputError("dialogue must have named turns and one missing reply")
            texts = [turn.text for turn in content if turn.text is not None]
            marked = [bool(parse_marked_example(text)[1]) for text in texts]
            if target_placement == TargetPlacement.DIALOGUE and not any(marked):
                raise InvalidOutputError("dialogue does not mark the target")
            if target_placement == TargetPlacement.OPTIONS:
                if any(marked):
                    raise InvalidOutputError(
                        "dialogue must hide the target when placement is options"
                    )
                if any(not parse_marked_example(option.content)[1] for option in options):
                    raise InvalidOutputError("every option must mark the target")
            if any("_" in text for text in texts):
                raise InvalidOutputError("dialogue must use null rather than a blank")
            continue
        if not isinstance(content, str) or not content.strip() or _INSTRUCTION.match(content):
            raise InvalidOutputError("invalid learner-facing Question content")
        if kind == QuestionType.CLOZE_TO_WORD and content.count("_") != 1:
            raise InvalidOutputError("cloze requires exactly one full-answer blank")
        spans = parse_marked_example(content)[1]
        if kind in {
            QuestionType.WORD_TO_DEFINITION,
            QuestionType.WORD_TO_USAGE,
            QuestionType.MEANING_IN_CONTEXT,
        }:
            if not spans:
                raise InvalidOutputError("target surface is not marked")


def _components(surface: str) -> tuple[str, ...]:
    """Keep fixed components ordered, excluding citation slots, not their literal counterparts."""
    return tuple(answer_key(re.sub(r"\{[^{}]+\}", " ", surface)).split())


def _target_sequences(word: Word, sense: Sense) -> set[tuple[str, ...]]:
    sequences = {_components(surface) for surface in [word.lemma, *word.aliases]}
    base = _components(word.lemma)
    for form in sense.forms:
        components = _components(form.surface)
        sequences.add(
            components if len(base) == 1 or len(components) > 1 else components + base[1:]
        )
    return sequences


def _validate_marked_target(text: str, sequences: set[tuple[str, ...]], *, required: bool):
    _, spans = parse_marked_example(text)
    if not spans:
        if required:
            raise InvalidOutputError("target expression is not marked")
        return
    tokens = tuple(token for span in spans for token in answer_key(span.surface).split())
    # Multiple occurrences are allowed, but each must preserve all fixed components in order.
    positions = {0}
    for start in range(len(tokens)):
        if start in positions:
            for sequence in sequences:
                if sequence and tokens[start : start + len(sequence)] == sequence:
                    positions.add(start + len(sequence))
    if len(tokens) not in positions:
        raise InvalidOutputError("tags do not mark a complete licensed target expression")


def _reveals_target(text: str, sequences: set[tuple[str, ...]]) -> bool:
    text = answer_key(text)
    return any(
        re.search(r"(?<!\w)" + r"\s+".join(map(re.escape, sequence)) + r"(?!\w)", text)
        for sequence in sequences
        if sequence
    )


def validate_targets(
    question: GeneratedQuestion,
    kind: QuestionType,
    word: Word,
    sense: Sense,
    *,
    target_placement: TargetPlacement | None,
) -> None:
    sequences = _target_sequences(word, sense)
    if kind == QuestionType.DIALOGUE_COMPLETION:
        for turn in question.content:
            if turn.text is None:
                continue
            if target_placement == TargetPlacement.OPTIONS and _reveals_target(
                turn.text, sequences
            ):
                raise InvalidOutputError("dialogue reveals the target when placement is options")
            _validate_marked_target(turn.text, sequences, required=False)
        for option in [question.correct, *question.distractors]:
            _validate_marked_target(
                option.content, sequences, required=target_placement == TargetPlacement.OPTIONS
            )
    if kind == QuestionType.WORD_TO_DEFINITION:
        valid = {answer_key(sense.definition.content)} if sense.definition is not None else set()
        if any(answer_key(option.content) in valid for option in question.distractors):
            raise InvalidOutputError("distractor repeats a trusted definition")
    if kind in {QuestionType.DEFINITION_TO_WORD, QuestionType.CONTEXT_TO_WORD}:
        if any(_components(option.content) in sequences for option in question.distractors):
            raise InvalidOutputError("distractor repeats a licensed target expression")
