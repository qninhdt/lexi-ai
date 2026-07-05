"""The seed format plugins + registry wiring.

Each plugin is self-contained: it owns ``generate`` (Entry -> Questions) and
``grade`` ((Question, answer) -> Score) for exactly one format, delegating the
grading logic to a shared helper in :mod:`lexi_ai.questions.scoring`. A plugin
that wants persistence calls ``ctx.store.insert(...)`` itself, inside ``generate``
— the engine never learns which plugins persist.

Importing this module runs the ``register(...)`` calls at the bottom, so the
registry is populated on package import.
"""

import random
import re

from langchain_core.messages import HumanMessage, SystemMessage

from lexi_ai.normalize import match_key
from lexi_ai.questions.base import (
    FormatSpec,
    QuestionContext,
    register,
)
from lexi_ai.llm import ainvoke_structured
from lexi_ai.questions.schemas import (
    ClozePayload,
    GeneratedMCQ,
    MCQPayload,
    UseInSentencePayload,
)
from lexi_ai.questions.scoring import grade_rubric, grade_single_choice, grade_text_span
from lexi_ai.read_models import Entry, Question, Score, SenseView

# Target option count for MCQs (1 correct + 3 distractors); degrade below this
# when distractor sources are thin, down to a floor of one distractor.
_MCQ_OPTIONS = 4
_MCQ_MIN_DISTRACTORS = 1

from lexi_ai.prompts import PromptLoader

_CONTEXTUAL_SYSTEM = PromptLoader.render("contextual_mcq_system")


def _core_sense(entry: Entry) -> SenseView | None:
    """The entry's core sense (senses are core-first), or None if it has none."""
    return entry.senses[0] if entry.senses else None


def _shuffled_options(correct: str, distractors: list[str], seed: str) -> tuple[list[str], int]:
    """Interleave correct + distractors in a deterministic, seed-stable order.

    Uses a LOCAL ``random.Random`` (no global RNG state) so option order is stable
    across runs and testable, satisfying the suite's no-nondeterminism rule.
    """
    options = [correct, *distractors]
    random.Random(seed).shuffle(options)
    return options, options.index(correct)


def _mcq_question(entry: Entry, sense: SenseView, stem: str, seed: str, distractors: list[str]):
    """Build a validated single-choice Question (or None if too few options)."""
    if len(distractors) < _MCQ_MIN_DISTRACTORS:
        return None
    options, correct_index = _shuffled_options(entry.display, distractors, seed)
    payload = MCQPayload(stem=stem, options=options, correct_index=correct_index)
    return Question(
        id=None,
        word_id=entry.word_id,
        sense_id=sense.sense_id,
        format="",  # set by the caller (the plugin knows its own id)
        answer_kind="single_choice",
        payload=payload.model_dump(),
    )


class DefinitionMCQ:
    """Rule MCQ: 'which word means <definition>?' — ephemeral (not persisted)."""

    format = "definition_mcq"
    answer_kind = "single_choice"

    async def generate(self, ctx: QuestionContext, n: int = 1) -> list[Question]:
        entry = ctx.entry
        if entry is None or n <= 0:
            return []
        sense = _core_sense(entry)
        if sense is None:
            return []
        distractors = await ctx.distractors.for_word(entry, k=_MCQ_OPTIONS - 1, pos=sense.pos)
        stem = f"Which word means: {sense.definition}"
        seed = f"definition_mcq:{match_key(entry.norm)}"
        q = _mcq_question(entry, sense, stem, seed, distractors)
        if q is None:
            return []
        q.format = self.format
        return [q]

    async def grade(self, ctx: QuestionContext, question: Question, answer: object) -> Score:
        return await grade_single_choice(question, answer)


class Cloze:
    """Rule fill-in-the-blank from a sense example — ephemeral."""

    format = "cloze"
    answer_kind = "text_span"

    async def generate(self, ctx: QuestionContext, n: int = 1) -> list[Question]:
        entry = ctx.entry
        if entry is None or n <= 0:
            return []
        sense = _core_sense(entry)
        if sense is None:
            return []
        for example in sense.examples:
            blanked = _blank_target(example, entry)
            if blanked is None:
                continue  # target not locatable in this example — try the next
            payload = ClozePayload(stem_with_blank=blanked, answer_norm=entry.norm)
            return [
                Question(
                    id=None,
                    word_id=entry.word_id,
                    sense_id=sense.sense_id,
                    format=self.format,
                    answer_kind=self.answer_kind,
                    payload=payload.model_dump(),
                )
            ]
        return []  # no example contained the target

    async def grade(self, ctx: QuestionContext, question: Question, answer: object) -> Score:
        return await grade_text_span(question, answer)


class ContextualMCQ:
    """LLM MCQ from a novel context — PERSISTS its output via ``ctx.store``."""

    format = "contextual_mcq"
    answer_kind = "single_choice"

    async def generate(self, ctx: QuestionContext, n: int = 1) -> list[Question]:
        entry = ctx.entry
        if entry is None or n <= 0 or ctx.llm is None:
            return []  # no llm configured -> this format is unavailable, best-effort
        sense = _core_sense(entry)
        if sense is None:
            return []
        human = PromptLoader.render(
            "contextual_mcq_user",
            word=entry.display,
            definition=sense.definition,
        )
        mcq = await ainvoke_structured(
            ctx.llm,
            [SystemMessage(content=_CONTEXTUAL_SYSTEM), HumanMessage(content=human)],
            GeneratedMCQ,
        )
        # We use only the llm's stem + distractors; the correct answer is always the
        # target word (entry.display), NOT the model's claimed `mcq.correct` — trusting
        # a generated answer would risk a hallucinated key. `correct` steers the model
        # to build coherent distractors, then is intentionally discarded.
        distractors = await self._merge_distractors(ctx, entry, sense, mcq)
        q = _mcq_question(
            entry, sense, mcq.stem, f"contextual_mcq:{match_key(entry.norm)}", distractors
        )
        if q is None:
            return []
        q.format = self.format
        if ctx.store is not None:  # the plugin decides to persist; the engine does not
            q = await ctx.store.insert(q)
        return [q]

    @staticmethod
    async def _merge_distractors(
        ctx: QuestionContext, entry: Entry, sense: SenseView, mcq: GeneratedMCQ
    ) -> list[str]:
        """LLM-proposed distractors first (answer-filtered), topped up from the ladder."""
        exclude = {match_key(entry.norm)}
        exclude.update(match_key(a.alias_norm) for a in entry.aliases)
        merged: list[str] = []
        seen: set[str] = set()
        for cand in mcq.distractors:
            key = match_key(cand)
            if key and key not in exclude and key not in seen:
                seen.add(key)
                merged.append(cand)
        if len(merged) < _MCQ_OPTIONS - 1:
            for cand in await ctx.distractors.for_word(entry, k=_MCQ_OPTIONS - 1, pos=sense.pos):
                key = match_key(cand)
                if key not in seen:
                    seen.add(key)
                    merged.append(cand)
                if len(merged) >= _MCQ_OPTIONS - 1:
                    break
        return merged[: _MCQ_OPTIONS - 1]

    async def grade(self, ctx: QuestionContext, question: Question, answer: object) -> Score:
        # Same helper the rule DefinitionMCQ uses — the cross-axis proof.
        return await grade_single_choice(question, answer)


class UseInSentence:
    """Rule prompt to use the word in a sentence — graded by the llm rubric judge."""

    format = "use_in_sentence"
    answer_kind = "free_text"

    async def generate(self, ctx: QuestionContext, n: int = 1) -> list[Question]:
        entry = ctx.entry
        if entry is None or n <= 0:
            return []
        sense = _core_sense(entry)
        if sense is None:
            return []
        payload = UseInSentencePayload(
            prompt=f"Write a sentence using '{entry.display}' to mean: {sense.definition}",
            target_norm=entry.norm,
            rubric=(
                f"Sentence must use '{entry.display}' with the sense: "
                f"'{sense.definition}'; grammatical; at least 6 words."
            ),
        )
        return [
            Question(
                id=None,
                word_id=entry.word_id,
                sense_id=sense.sense_id,
                format=self.format,
                answer_kind=self.answer_kind,
                payload=payload.model_dump(),
            )
        ]

    async def grade(self, ctx: QuestionContext, question: Question, answer: object) -> Score:
        return await grade_rubric(question, answer, judge=ctx.judge)


def _blank_target(example: str, entry: Entry) -> str | None:
    """Replace the target's surface in ``example`` with a blank, or None if absent.

    Tries a **word-boundary** whole-phrase match first (handles multi-word idioms
    and, crucially, avoids blanking the target inside a longer word — ``eloquent``
    must not fire on ``eloquently``, ``cat`` not on ``category``). Then falls back
    token-by-token, folding each token through ``match_key`` so a diacritic/case
    variant still matches. A truly inflected surface (``runs`` vs ``run``) folds
    to a different key, so it is skipped and the caller tries the next example —
    the documented ``match_key`` limitation.
    """
    blank = "_____"
    target_key = match_key(entry.norm)
    display_lower = entry.display.lower()
    # Whole-phrase, case-insensitive, anchored to word boundaries so a substring
    # of a longer word (or an inflected form) is NOT matched. Lookarounds instead
    # of \b so a target that starts/ends with a non-word char still behaves.
    m = re.search(rf"(?<!\w){re.escape(display_lower)}(?!\w)", example.lower())
    if m:
        return example[: m.start()] + blank + example[m.end() :]
    # Token-by-token via match_key equality (catches diacritic/case variants the
    # boundary match missed because the surface differs from the display).
    tokens = example.split()
    for i, tok in enumerate(tokens):
        stripped = tok.strip(".,;:!?\"'()[]")
        if stripped and match_key(stripped) == target_key:
            tokens[i] = tok.replace(stripped, blank)
            return " ".join(tokens)
    return None


# --- seed wiring: one line per format (this IS "add a format cheaply") --------

register(FormatSpec("definition_mcq", "single_choice", DefinitionMCQ))
register(FormatSpec("cloze", "text_span", Cloze))
register(FormatSpec("contextual_mcq", "single_choice", ContextualMCQ))
register(FormatSpec("use_in_sentence", "free_text", UseInSentence))
