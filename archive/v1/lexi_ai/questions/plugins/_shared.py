"""Cross-type helpers shared by the per-type plugin modules.

MCQ assembly, the deterministic shuffle, and blank-the-target live here so no type module imports
another. Builders return the internal :class:`PersistedQuestion` DRAFT carrier
(``question_id=None``); the pydantic payload validators still run first so a bad index or empty
option can never reach the persistence boundary.
"""

import random
import re

from lexi_ai.models import Entry, SenseView
from lexi_ai.prompts import PromptLoader
from lexi_ai.questions.schemas import FlashcardPayload, MCQPayload
from lexi_ai.questions.types import PersistedQuestion, RenderKind
from lexi_ai.text import answer_key, parse_marked_example

# Target option count for MCQs (1 correct + 3 distractors); degrade below this
# when distractor sources are thin, down to a floor of one distractor.
_MCQ_OPTIONS = 4
_MCQ_MIN_DISTRACTORS = 1

# Rendered once at import; the sole llm-generated type (contextual_mcq) reuses it.
_CONTEXTUAL_SYSTEM = PromptLoader.render("contextual_mcq_system")


def _accepted_forms(sense: SenseView) -> list[str]:
    """Inflected surfaces the text-span grader folds equal to the answer, so a learner typing
    ``ran`` for ``run`` scores right. Empty when the sense has no forms — grading then falls back
    to the lemma norm alone.
    """
    return [f.surface for f in sense.forms if f.surface]


def _shuffled_options(correct: str, distractors: list[str], seed: str) -> tuple[list[str], int]:
    """Interleave correct + distractors in a deterministic, seed-stable order.

    A LOCAL ``random.Random`` (no global RNG state) keeps option order stable across runs and
    testable.
    """
    options = [correct, *distractors]
    random.Random(seed).shuffle(options)
    return options, options.index(correct)


def _mcq_question(
    entry: Entry,
    sense: SenseView,
    stem: str,
    seed: str,
    distractors: list[str],
    *,
    type_id: str,
    difficulty_level: int,
) -> PersistedQuestion | None:
    """Build a validated assessment draft, or ``None`` if options are thin."""
    if len(distractors) < _MCQ_MIN_DISTRACTORS:
        return None
    options, correct_index = _shuffled_options(entry.display, distractors, seed)
    payload = MCQPayload(stem=stem, options=options, correct_index=correct_index)
    return PersistedQuestion(
        question_id=None,
        word_id=entry.word_id,
        sense_id=sense.sense_id,
        type_id=type_id,
        render_kind=RenderKind.SINGLE_CHOICE,
        difficulty_level=difficulty_level,
        interaction="assessment",
        payload=payload.model_dump(),
    )


def _exposure_question(entry: Entry, sense: SenseView) -> PersistedQuestion:
    """Build a deterministic level-0 flashcard from authoritative sense data."""
    payload = FlashcardPayload(
        word=entry.display,
        pos=sense.pos or entry.pos,
        definition=sense.definition,
        example=sense.examples[0] if sense.examples else None,
        ipa_uk=sense.ipa_uk,
        ipa_us=sense.ipa_us,
    )
    return PersistedQuestion(
        question_id=None,
        word_id=entry.word_id,
        sense_id=sense.sense_id,
        type_id="flashcard",
        render_kind=RenderKind.FLASHCARD,
        difficulty_level=0,
        interaction="exposure",
        payload=payload.model_dump(),
    )


_BLANK = "_____"
# Punctuation stripped off a token's edges before its core is answer_key-compared.
_EDGE_PUNCT = ".,;:!?\"'()[]"


def _blank_first_token_matching(text: str, want: set[str]) -> str | None:
    """Blank the FIRST whitespace token whose (edge-punct-stripped) core folds into ``want`` (a set
    of ``answer_key`` values), preserving all original separators.

    - **Separators preserved:** tokens are spliced at their located offset, so
    double spaces / newlines survive into the stem (``" ".join(split())`` collapses them).
    - **Matched slice only:** ONLY the token's stripped core is replaced, leaving
    its edge punctuation — a token like ``cat-cat`` blanks the matched half.

    Folding rides on ``answer_key``, which lowercases without a length-changing ``.lower()`` (e.g.
    ``İ``) that would misalign the splice offsets. It also folds accents, which ``match_key``
    deliberately does not: that has to keep ``café`` and ``cafe`` available as separate headwords.
    """
    for m in re.finditer(r"\S+", text):
        tok = m.group(0)
        core = tok.strip(_EDGE_PUNCT)
        if not core or answer_key(core) not in want:
            continue
        blanked_tok = tok.replace(core, _BLANK, 1)
        return text[: m.start()] + blanked_tok + text[m.end() :]
    return None


def _blank_target(example: str, entry: Entry) -> str | None:
    """Replace the target's surface in ``example`` with a blank, or None if absent.

    Operates on the RENDERED (tag-free) text so ``<t inf>`` markup never leaks into the stem. A
    tagged span wins: the tag marks exactly the (possibly inflected) target surface, closing the
    inflection limitation (``ran`` for ``run``) the key-equality fallback cannot.

    Without a tag, falls back to a **word-boundary** whole-phrase match (``eloquent`` must not fire
    on ``eloquently``), then token-by-token ``answer_key`` equality for diacritic/case variants. A
    truly inflected untagged surface folds to a different key and is skipped — the caller tries the
    next example.
    """
    clean, spans = parse_marked_example(example)
    if spans:
        s = spans[0]
        return clean[: s.start] + _BLANK + clean[s.end :]
    display = entry.display
    # Lookarounds instead of \b so a target starting/ending with a non-word char still behaves.
    # IGNORECASE on the ORIGINAL clean: a length-changing .lower() (e.g. İ) would misalign offsets.
    m = re.search(rf"(?<!\w){re.escape(display)}(?!\w)", clean, flags=re.IGNORECASE)
    if m:
        return clean[: m.start()] + _BLANK + clean[m.end() :]
    # answer_key rather than match_key: match_key keeps `café` and `cafe` apart so they can be
    # separate headwords, which would leave an accented target unblanked.
    return _blank_first_token_matching(clean, {answer_key(entry.norm)})
