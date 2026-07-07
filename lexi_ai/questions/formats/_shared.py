"""Cross-format helpers shared by the per-format plugin modules.

Each format lives in its own module (``definition_mcq.py``, ``cloze.py``, ...);
the small amount of logic MORE than one format needs — MCQ assembly, the
deterministic shuffle, blank-the-target, the audio-ref resolver — lives here so
no format module imports another. Nothing in this module registers a format;
the plugin modules do that themselves on import (see ``__init__``).
"""

import random
import re

from lexi_ai.markup import parse_marked_example
from lexi_ai.normalize import match_key
from lexi_ai.prompts import PromptLoader
from lexi_ai.questions.base import QuestionContext
from lexi_ai.questions.schemas import AudioRef, MCQPayload
from lexi_ai.read_models import Entry, Question, SenseView

# Target option count for MCQs (1 correct + 3 distractors); degrade below this
# when distractor sources are thin, down to a floor of one distractor.
_MCQ_OPTIONS = 4
_MCQ_MIN_DISTRACTORS = 1

# Rendered once at import; the sole llm-generated format (contextual_mcq) reuses it.
_CONTEXTUAL_SYSTEM = PromptLoader.render("contextual_mcq_system")


def _core_sense(entry: Entry) -> SenseView | None:
    """The entry's core sense (senses are core-first), or None if it has none."""
    return entry.senses[0] if entry.senses else None


def _accepted_forms(sense: SenseView) -> list[str]:
    """Inflected surfaces the text-span grader folds equal to the answer, so a
    learner typing ``ran`` for ``run`` scores right. Empty when the sense has no
    forms — grading then falls back to the lemma norm alone."""
    return [f.surface for f in sense.forms if f.surface]


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


def _blank_target(example: str, entry: Entry) -> str | None:
    """Replace the target's surface in ``example`` with a blank, or None if absent.

    Operates on the RENDERED (tag-free) text so ``<t inf>`` markup never leaks into
    the stem. A tagged span wins: the tag marks exactly the (possibly inflected)
    target surface, so it is blanked directly — this closes the inflection
    limitation (``ran`` for ``run``) the pure ``match_key`` fallback cannot.

    Without a tag, falls back to a **word-boundary** whole-phrase match (avoids
    blanking the target inside a longer word — ``eloquent`` must not fire on
    ``eloquently``), then token-by-token ``match_key`` equality for diacritic/case
    variants. A truly inflected untagged surface still folds to a different key and
    is skipped — the caller tries the next example.
    """
    blank = "_____"
    clean, spans = parse_marked_example(example)
    if spans:
        s = spans[0]
        return clean[: s.start] + blank + clean[s.end :]
    target_key = match_key(entry.norm)
    display_lower = entry.display.lower()
    # Whole-phrase, case-insensitive, anchored to word boundaries so a substring
    # of a longer word (or an inflected form) is NOT matched. Lookarounds instead
    # of \b so a target that starts/ends with a non-word char still behaves.
    m = re.search(rf"(?<!\w){re.escape(display_lower)}(?!\w)", clean.lower())
    if m:
        return clean[: m.start()] + blank + clean[m.end() :]
    # Token-by-token via match_key equality (catches diacritic/case variants the
    # boundary match missed because the surface differs from the display).
    tokens = clean.split()
    for i, tok in enumerate(tokens):
        stripped = tok.strip(".,;:!?\"'()[]")
        if stripped and match_key(stripped) == target_key:
            tokens[i] = tok.replace(stripped, blank)
            return " ".join(tokens)
    return None


def _blank_in_phrase(phrase: str, entry: Entry, accepted: list[str]) -> str | None:
    """Blank the target token in a collocation, or None if it is not present.

    A collocation carries NO ``<t inf>`` markup (it is a plain partner phrase), so
    the target is found token-by-token via ``match_key`` equality against the lemma
    OR any accepted inflected surface — ``heavy rains`` blanks on the ``rains``
    form of ``rain``. Blanks the FIRST matching token; None when none matches (the
    caller tries the next collocation)."""
    blank = "_____"
    want = {match_key(entry.norm)}
    want.update(match_key(s) for s in accepted if s)
    tokens = phrase.split()
    for i, tok in enumerate(tokens):
        stripped = tok.strip(".,;:!?\"'()[]")
        if stripped and match_key(stripped) in want:
            tokens[i] = tok.replace(stripped, blank)
            return " ".join(tokens)
    return None


def _shuffled_pairs(definitions: list[str], seed: str) -> tuple[list[str], list[int]]:
    """Shuffle ``definitions`` deterministically; return (shuffled, correct_map).

    ``correct_map[i]`` is the position of the i-th original definition within the
    shuffled column, so left ``i`` maps to ``shuffled[correct_map[i]]``. Uses a
    LOCAL ``random.Random`` — no global RNG state, seed-stable across runs.
    """
    order = list(range(len(definitions)))
    random.Random(seed).shuffle(order)
    shuffled = [definitions[j] for j in order]
    # order[k] = original index now at shuffled position k; invert to map orig->pos.
    pos_of_original = [0] * len(order)
    for pos, orig in enumerate(order):
        pos_of_original[orig] = pos
    return shuffled, pos_of_original


async def _core_audio_ref(ctx: QuestionContext, sense: SenseView) -> AudioRef | None:
    """Ensure a TTS clip exists for the sense's definition and return its reference
    tuple, or ``None`` when TTS is unavailable / the clip can't be made.

    The audio source is the sense DEFINITION (``sense_def``) — the learner hears the
    meaning spoken and supplies the word — so it keys on ``sense.sense_id``. Requires
    a ``ctx.tts`` port (``None`` → format unavailable, like ``ctx.llm`` for the
    contextual MCQ) and a persisted sense id to address the clip.
    """
    if ctx.tts is None or sense.sense_id is None:
        return None
    ref = await ctx.tts.ensure_clip("sense_def", sense.sense_id)
    if ref is None:
        return None  # source text gone / clip couldn't be made — no fabricated asset
    source_kind, source_id, voice, fmt = ref
    return AudioRef(source_kind=source_kind, source_id=source_id, voice=voice, fmt=fmt)
