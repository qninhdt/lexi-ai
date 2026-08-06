"""Cross-type helpers shared by the per-type plugin modules.

Each question type lives in its own module (``definition_mcq.py``, ``cloze.py``,
...); the small amount of logic MORE than one type needs — MCQ assembly, the
deterministic shuffle, blank-the-target — lives here so no type module imports
another. Builders return the internal :class:`PersistedQuestion` DRAFT carrier
(``question_id=None``); the pydantic payload validators still run first so a bad
index or empty option can never reach the persistence boundary.
"""

import random
import re

from lexi_ai.contracts.questions import RenderKind
from lexi_ai.domain.questions import PersistedQuestion
from lexi_ai.markup import parse_marked_example
from lexi_ai.normalize import answer_key
from lexi_ai.prompts import PromptLoader
from lexi_ai.questions.schemas import FlashcardPayload, MCQPayload
from lexi_ai.read_models import Entry, SenseView

# Target option count for MCQs (1 correct + 3 distractors); degrade below this
# when distractor sources are thin, down to a floor of one distractor.
_MCQ_OPTIONS = 4
_MCQ_MIN_DISTRACTORS = 1

# Rendered once at import; the sole llm-generated type (contextual_mcq) reuses it.
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
    """Blank the FIRST whitespace token whose (edge-punct-stripped) core folds into
    ``want`` (a set of ``answer_key`` values), preserving all original separators.

    Shared by ``_blank_target``'s fallback and ``_blank_in_phrase``. Two
    correctness fixes over the earlier ``text.split()`` + ``" ".join()`` copies:

    - **Separators preserved:** iterate ``\\S+`` tokens by their located offset
      and splice the blank at that offset, so double spaces / newlines in the
      original survive into the stem (``" ".join(split())`` collapsed them).
    - **Matched slice only:** replace ONLY the token's stripped core, leaving
      its edge punctuation — and a token like ``cat-cat`` blanks the matched half,
      not the whole token (the core is what folds, so only it is replaced).

    Case-insensitive folding rides on ``answer_key`` (which lowercases), so no
    length-changing ``.lower()`` is applied to the text here — that was the
    slice-misalignment bug in the old boundary path. ``answer_key`` also folds
    accents, which both callers want and which ``match_key`` deliberately does
    not: it has to keep ``café`` and ``cafe`` available as separate headwords.
    Both callers build ``want`` with the same function, so the two sides agree.
    """
    for m in re.finditer(r"\S+", text):
        tok = m.group(0)
        core = tok.strip(_EDGE_PUNCT)
        if not core or answer_key(core) not in want:
            continue
        # Blank only the core within the token, then splice at the token offset so
        # every separator (runs of spaces, newlines) outside the token is untouched.
        blanked_tok = tok.replace(core, _BLANK, 1)
        return text[: m.start()] + blanked_tok + text[m.end() :]
    return None


def _blank_target(example: str, entry: Entry) -> str | None:
    """Replace the target's surface in ``example`` with a blank, or None if absent.

    Operates on the RENDERED (tag-free) text so ``<t inf>`` markup never leaks into
    the stem. A tagged span wins: the tag marks exactly the (possibly inflected)
    target surface, so it is blanked directly — this closes the inflection
    limitation (``ran`` for ``run``) the pure key-equality fallback cannot.

    Without a tag, falls back to a **word-boundary** whole-phrase match (avoids
    blanking the target inside a longer word — ``eloquent`` must not fire on
    ``eloquently``), then token-by-token ``answer_key`` equality for diacritic/case
    variants. A truly inflected untagged surface still folds to a different key and
    is skipped — the caller tries the next example.
    """
    clean, spans = parse_marked_example(example)
    if spans:
        s = spans[0]
        return clean[: s.start] + _BLANK + clean[s.end :]
    display = entry.display
    # Whole-phrase, case-insensitive, anchored to word boundaries so a substring
    # of a longer word (or an inflected form) is NOT matched. Lookarounds instead
    # of \b so a target that starts/ends with a non-word char still behaves.
    # re.IGNORECASE on the ORIGINAL clean (no pre-lowercasing): a length-changing
    # .lower() (e.g. İ -> i̇) would misalign the match offsets against clean.
    m = re.search(rf"(?<!\w){re.escape(display)}(?!\w)", clean, flags=re.IGNORECASE)
    if m:
        return clean[: m.start()] + _BLANK + clean[m.end() :]
    # Token-by-token via answer_key equality (catches diacritic/case variants the
    # boundary match missed because the surface differs from the display).
    # answer_key rather than match_key: matching here is the whole point, and
    # match_key deliberately keeps `café` and `cafe` apart so they can be separate
    # headwords — which would leave an accented target unblanked.
    return _blank_first_token_matching(clean, {answer_key(entry.norm)})


def _blank_in_phrase(phrase: str, entry: Entry, accepted: list[str]) -> str | None:
    """Blank the target token in a collocation, or None if it is not present.

    A collocation carries NO ``<t inf>`` markup (it is a plain partner phrase), so
    the target is found token-by-token via ``answer_key`` equality against the lemma
    OR any accepted inflected surface — ``heavy rains`` blanks on the ``rains``
    form of ``rain``. Blanks the FIRST matching token; None when none matches (the
    caller tries the next collocation)."""
    want = {answer_key(entry.norm)}
    want.update(answer_key(s) for s in accepted if s)
    return _blank_first_token_matching(phrase, want)
