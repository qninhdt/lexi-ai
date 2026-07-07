"""Tests for the questions subsystem — the acceptance surface for the plan.

No live LLM, no network: the contextual-MCQ generator and the rubric judge are
fake ``StructuredLLM``s returning canned schema objects, matching the existing
suite's posture (see ``tests/test_generation.py``). In-memory SQLite with a shared
connection (StaticPool) backs persistence, like ``tests/test_repository.py``.

Covers the six groups from the plan: plugin generation, grade helpers, the
cross-axis proofs, plugin-owned persistence (engine-blind), registry validation
and one-line extensibility, plus payload round-trip edges.
"""

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from lexi_ai.db import create_session_factory, init_models, session_scope
from lexi_ai.models import Question as QuestionRow
from lexi_ai.models import Sense, Word
from lexi_ai.questions.base import REGISTRY, FormatSpec, QuestionContext, register
from lexi_ai.questions.engine import QuestionEngine
from lexi_ai.questions.formats import (
    Cloze,
    CollocationFill,
    ContextualMCQ,
    DefinitionMCQ,
    Listening,
    Matching,
    PronunciationMCQ,
    Spelling,
    UseInSentence,
)
from lexi_ai.questions.repository import QuestionRepository
from lexi_ai.questions.schemas import GeneratedMCQ, Judgment
from lexi_ai.questions.scoring import grade_matching, grade_single_choice, grade_text_span
from lexi_ai.read_models import Entry, FormView, Question, SenseView

# --- fixtures + helpers ---------------------------------------------------


@pytest.fixture
async def session_factory():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _fk_on(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    await init_models(engine)
    yield create_session_factory(engine)
    await engine.dispose()


async def _seed_word(
    session_factory, norm="eloquent", definition="fluent and persuasive in speech"
) -> tuple[int, int]:
    """Insert a done word + one core sense; return (word_id, sense_id)."""
    async with session_scope(session_factory) as session:
        word = Word(norm=norm, match_key=norm, entry_type="word", status="done", pos="adj")
        session.add(word)
        await session.flush()
        sense = Sense(word_id=word.id, definition=definition, tier="core", pos="adj")
        session.add(sense)
        await session.flush()
        return word.id, sense.id


def _entry(
    word_id,
    sense_id,
    *,
    norm="eloquent",
    examples=None,
    status="done",
    forms=None,
    ipa_uk=None,
    ipa_us=None,
    collocations=None,
) -> Entry:
    return Entry(
        display=norm,
        norm=norm,
        entry_type="word",
        pos="adj",
        status=status,
        word_id=word_id,
        senses=[
            SenseView(
                definition="fluent and persuasive in speech",
                tier="core",
                pos="adj",
                cefr_level="C1",
                ipa_uk=ipa_uk,
                ipa_us=ipa_us,
                examples=examples if examples is not None else ["She gave an eloquent speech."],
                forms=forms if forms is not None else [],
                collocations=collocations if collocations is not None else [],
                sense_id=sense_id,
            )
        ],
    )


class FakeDistractors:
    """A DistractorProvider stand-in returning a fixed pool (no DB/embedder)."""

    def __init__(self, pool=("terse", "clumsy", "dull", "vague")):
        self._pool = list(pool)

    async def for_word(self, entry, *, k, pos=None):
        return self._pool[:k]


class _FakeLLM:
    """Minimal ``StructuredLLM``: returns a canned object (or raises) per ``parse``,
    ignoring the schema — the callback already returns the right type."""

    def __init__(self, fn):
        self._fn = fn

    async def parse(self, messages, schema):
        return await self._fn(messages)


def _fake_llm(obj):
    async def _call(_messages):
        return obj

    return _FakeLLM(_call)


def _mcq_ctx(entry, **kw):
    return QuestionContext(entry=entry, distractors=FakeDistractors(), **kw)


class FakeTtsPort:
    """A TtsPort stand-in: records the clips it was asked to ensure and returns a
    reference tuple, no synthesis/network. ``missing`` returns ``None`` (the source
    text was gone) so the plugin's no-clip degrade path is exercised."""

    def __init__(self, *, voice="alloy", fmt="mp3", missing=False):
        self._voice = voice
        self._fmt = fmt
        self._missing = missing
        self.calls: list[tuple[str, int]] = []

    async def ensure_clip(self, source_kind, source_id):
        self.calls.append((source_kind, source_id))
        if self._missing:
            return None
        return (source_kind, source_id, self._voice, self._fmt)


async def _count(session_factory) -> int:
    async with session_scope(session_factory) as session:
        return (await session.execute(select(func.count()).select_from(QuestionRow))).scalar_one()


# --- 1. plugin generate ---------------------------------------------------


async def test_definition_mcq_builds_valid_four_option_question(session_factory):
    wid, sid = await _seed_word(session_factory)
    qs = await DefinitionMCQ().generate(_mcq_ctx(_entry(wid, sid)), n=1)
    assert len(qs) == 1
    payload = qs[0].payload
    assert len(payload["options"]) == 4
    # exactly one correct, and it is the target display
    assert payload["options"][payload["correct_index"]] == "eloquent"
    # distractors exclude the answer (match_key exclusion)
    assert payload["options"].count("eloquent") == 1
    assert qs[0].sense_id == sid  # core-sense provenance


async def test_definition_mcq_degrades_when_no_distractors(session_factory):
    wid, sid = await _seed_word(session_factory)
    ctx = QuestionContext(entry=_entry(wid, sid), distractors=FakeDistractors(pool=[]))
    # no distractor sources -> best-effort: no question, no raise
    qs = await DefinitionMCQ().generate(ctx, n=1)
    assert qs == []


async def test_n_is_best_effort_max_single_core_sense(session_factory):
    wid, sid = await _seed_word(session_factory)
    # n=3 but only one core sense -> one question, no fabrication
    qs = await DefinitionMCQ().generate(_mcq_ctx(_entry(wid, sid)), n=3)
    assert len(qs) == 1


async def test_cloze_blanks_target_and_stores_canonical_answer(session_factory):
    wid, sid = await _seed_word(session_factory)
    qs = await Cloze().generate(_mcq_ctx(_entry(wid, sid)), n=1)
    assert len(qs) == 1
    assert "eloquent" not in qs[0].payload["stem_with_blank"].lower()
    assert qs[0].payload["answer_norm"] == "eloquent"


async def test_cloze_skips_examples_without_target_then_returns_empty(session_factory):
    wid, sid = await _seed_word(session_factory)
    # first example lacks the target, none contain it -> []
    entry = _entry(wid, sid, examples=["A totally unrelated line.", "Another one."])
    qs = await Cloze().generate(_mcq_ctx(entry), n=1)
    assert qs == []


async def test_cloze_picks_first_usable_example(session_factory):
    wid, sid = await _seed_word(session_factory)
    entry = _entry(wid, sid, examples=["No target here.", "An eloquent reply followed."])
    qs = await Cloze().generate(_mcq_ctx(entry), n=1)
    assert len(qs) == 1
    assert "_____" in qs[0].payload["stem_with_blank"]
    assert "eloquent" not in qs[0].payload["stem_with_blank"].lower()


async def test_cloze_does_not_blank_inside_longer_word(session_factory):
    # "eloquently" must NOT be blanked as "_____ly"; the target only stands alone
    # in the second example, so that is the one used.
    wid, sid = await _seed_word(session_factory)
    entry = _entry(wid, sid, examples=["She spoke eloquently.", "An eloquent reply."])
    qs = await Cloze().generate(_mcq_ctx(entry), n=1)
    assert len(qs) == 1
    stem = qs[0].payload["stem_with_blank"]
    assert stem == "An _____ reply." and "ly" not in stem.replace("reply", "")


async def test_cloze_skips_when_target_only_appears_as_substring(session_factory):
    # Only inflected/substring occurrences -> no usable example -> [] (no broken blank).
    wid, sid = await _seed_word(session_factory, norm="cat", definition="a small feline")
    entry = _entry(wid, sid, norm="cat", examples=["The category was long.", "A catalog of cats."])
    qs = await Cloze().generate(_mcq_ctx(entry), n=1)
    # "cats" folds to a different match_key than "cat"; "category"/"catalog" are
    # substrings only — none is a standalone "cat", so cloze yields nothing here.
    assert qs == []


async def test_contextual_mcq_merges_and_filters_llm_distractors(session_factory):
    wid, sid = await _seed_word(session_factory)
    # llm proposes the correct answer inside its own distractors -> filtered out
    mcq = GeneratedMCQ(
        stem="His ____ speech moved us.", correct="eloquent", distractors=["eloquent", "terse"]
    )
    ctx = _mcq_ctx(_entry(wid, sid), llm=_fake_llm(mcq))
    qs = await ContextualMCQ().generate(ctx, n=1)
    assert len(qs) == 1
    opts = qs[0].payload["options"]
    assert len(opts) == 4 and opts.count("eloquent") == 1  # answer not duplicated as distractor
    assert opts[qs[0].payload["correct_index"]] == "eloquent"


async def test_contextual_mcq_without_llm_is_unavailable(session_factory):
    wid, sid = await _seed_word(session_factory)
    qs = await ContextualMCQ().generate(_mcq_ctx(_entry(wid, sid)), n=1)  # ctx.llm is None
    assert qs == []


async def test_use_in_sentence_payload_has_target_and_rubric(session_factory):
    wid, sid = await _seed_word(session_factory)
    qs = await UseInSentence().generate(_mcq_ctx(_entry(wid, sid)), n=1)
    assert len(qs) == 1
    assert qs[0].payload["target_norm"] == "eloquent"
    assert "eloquent" in qs[0].payload["rubric"]


async def test_generate_option_order_is_deterministic(session_factory):
    wid, sid = await _seed_word(session_factory)
    a = await DefinitionMCQ().generate(_mcq_ctx(_entry(wid, sid)), n=1)
    b = await DefinitionMCQ().generate(_mcq_ctx(_entry(wid, sid)), n=1)
    assert a[0].payload["options"] == b[0].payload["options"]  # seeded shuffle, stable


# --- 2. grade helpers -----------------------------------------------------


def _mcq_question(options, correct_index) -> Question:
    return Question(
        id=None,
        word_id=1,
        sense_id=None,
        format="definition_mcq",
        answer_kind="single_choice",
        payload={"stem": "s", "options": options, "correct_index": correct_index},
    )


async def test_grade_single_choice_by_index():
    q = _mcq_question(["a", "b", "c", "d"], 2)
    assert (await grade_single_choice(q, 2)).correct is True
    assert (await grade_single_choice(q, 2)).score == 1.0
    assert (await grade_single_choice(q, 0)).correct is False
    assert (await grade_single_choice(q, 0)).score == 0.0


async def test_grade_single_choice_by_value_matchkey():
    q = _mcq_question(["Colour", "shade", "hue", "tint"], 0)
    # answer by value, case-insensitive via match_key
    assert (await grade_single_choice(q, "colour")).correct is True
    assert (await grade_single_choice(q, "HUE")).correct is False


async def test_grade_single_choice_unmatched_is_wrong_not_error():
    q = _mcq_question(["a", "b"], 0)
    score = await grade_single_choice(q, "nonexistent-option")
    assert score.correct is False and score.kind == "rule"


async def test_grade_text_span_rides_the_one_normalizer():
    # contract test: grading folds exactly what match_key folds — case, whitespace,
    # and diacritics — but NOT spelling variants (colour != color; that is an
    # alias concern, not normalization). This proves cloze grading can never drift
    # from how the dictionary keys words.
    q = Question(
        id=None,
        word_id=1,
        sense_id=None,
        format="cloze",
        answer_kind="text_span",
        payload={"stem_with_blank": "_", "answer_norm": "café"},
    )
    assert (await grade_text_span(q, "cafe")).correct is True  # diacritic folds
    assert (await grade_text_span(q, "  CAFÉ ")).correct is True  # case + whitespace fold
    q2 = Question(
        id=None,
        word_id=1,
        sense_id=None,
        format="cloze",
        answer_kind="text_span",
        payload={"stem_with_blank": "_", "answer_norm": "color"},
    )
    assert (await grade_text_span(q2, "colour")).correct is False  # spelling is NOT folded


async def test_grade_text_span_accepts_inflected_forms():
    # accepted_forms lets a learner type an inflected surface (ran for run) and
    # score right, WITHOUT changing match_key — the surfaces come from the sense's
    # independent forms field, not from folding. This closes the documented
    # cloze/spelling limitation (runs != run).
    q = Question(
        id=None,
        word_id=1,
        sense_id=None,
        format="spelling",
        answer_kind="text_span",
        payload={"answer_norm": "run", "accepted_forms": ["ran", "running", "runs"]},
    )
    assert (await grade_text_span(q, "run")).correct is True  # the lemma
    assert (await grade_text_span(q, "ran")).correct is True  # past form
    assert (await grade_text_span(q, "RUNNING ")).correct is True  # form + case/ws fold
    assert (await grade_text_span(q, "swam")).correct is False  # unrelated word


async def test_cloze_carries_accepted_forms_from_sense(session_factory):
    wid, sid = await _seed_word(session_factory)
    forms = [FormView(inf="present_3sg", surface="runs"), FormView(inf="past", surface="ran")]
    entry = _entry(wid, sid, norm="run", examples=["They run daily."], forms=forms)
    qs = await Cloze().generate(_mcq_ctx(entry), n=1)
    assert len(qs) == 1
    assert qs[0].payload["accepted_forms"] == ["runs", "ran"]
    # and the plugin's own grade folds an inflected answer through that payload
    assert (await Cloze().grade(_mcq_ctx(None), qs[0], "ran")).correct is True


async def test_grade_rubric_maps_judgment_to_llm_score(session_factory):
    wid, sid = await _seed_word(session_factory)
    q = (await UseInSentence().generate(_mcq_ctx(_entry(wid, sid)), n=1))[0]
    judge = _fake_llm(Judgment(correct=True, score=0.75, feedback="Good."))
    ctx = QuestionContext(entry=None, distractors=FakeDistractors(), judge=judge)
    score = await UseInSentence().grade(ctx, q, "The eloquent speaker held the room.")
    assert score.kind == "llm" and score.correct is True
    assert score.score == 0.75 and score.feedback == "Good."


async def test_grade_rubric_raises_on_persistent_failure(session_factory):
    wid, sid = await _seed_word(session_factory)
    q = (await UseInSentence().generate(_mcq_ctx(_entry(wid, sid)), n=1))[0]

    async def _boom(_messages):
        raise RuntimeError("judge down")

    ctx = QuestionContext(entry=None, distractors=FakeDistractors(), judge=_FakeLLM(_boom))
    with pytest.raises(RuntimeError, match="judge down"):
        await UseInSentence().grade(ctx, q, "some answer")


# --- 3. cross-axis proofs -------------------------------------------------


async def test_llm_generated_question_graded_by_shared_rule_helper(session_factory):
    """contextual_mcq (llm-authored) grades through the same helper as the rule MCQ."""
    wid, sid = await _seed_word(session_factory)
    mcq = GeneratedMCQ(stem="His ____ speech.", correct="eloquent", distractors=["terse", "dull"])
    ctx = _mcq_ctx(_entry(wid, sid), llm=_fake_llm(mcq))
    q = (await ContextualMCQ().generate(ctx, n=1))[0]
    score = await ContextualMCQ().grade(ctx, q, q.payload["correct_index"])
    assert score.correct is True and score.kind == "rule"  # graded by rule despite llm origin


async def test_rule_generated_question_graded_by_llm_judge(session_factory):
    """use_in_sentence (rule-authored) grades through the llm rubric judge."""
    wid, sid = await _seed_word(session_factory)
    q = (await UseInSentence().generate(_mcq_ctx(_entry(wid, sid)), n=1))[0]
    judge = _fake_llm(Judgment(correct=True, score=1.0, feedback="Perfect."))
    ctx = QuestionContext(entry=None, distractors=FakeDistractors(), judge=judge)
    score = await UseInSentence().grade(ctx, q, "An eloquent orator swayed the crowd today.")
    assert score.kind == "llm" and score.correct is True  # rule-gen, llm-graded


# --- 4. persistence (plugin-owned, engine-blind) --------------------------


def _engine(session_factory, *, llm=None, judge=None) -> QuestionEngine:
    return QuestionEngine(
        QuestionRepository(session_factory),
        FakeDistractors(),
        llm=llm,
        judge_llm=judge,
    )


async def test_persisting_plugin_writes_row_and_lists(session_factory):
    wid, sid = await _seed_word(session_factory)
    mcq = GeneratedMCQ(stem="His ____ speech.", correct="eloquent", distractors=["terse", "dull"])
    eng = _engine(session_factory, llm=_fake_llm(mcq))
    qs = await eng.generate(_entry(wid, sid), formats=["contextual_mcq"], n=1)
    assert qs[0].id is not None
    assert await _count(session_factory) == 1
    listed = await eng.list(wid, fmt="contextual_mcq")
    assert len(listed) == 1
    assert listed[0].payload == qs[0].payload  # round-trips dict-equal
    assert listed[0].id == qs[0].id


async def test_ephemeral_plugins_leave_no_row(session_factory):
    wid, sid = await _seed_word(session_factory)
    eng = _engine(session_factory)
    qs = await eng.generate(
        _entry(wid, sid), formats=["definition_mcq", "cloze", "use_in_sentence"], n=1
    )
    assert all(q.id is None for q in qs)
    assert await _count(session_factory) == 0  # zero rows


async def test_generate_on_non_done_entry_raises(session_factory):
    wid, sid = await _seed_word(session_factory)
    eng = _engine(session_factory)
    with pytest.raises(ValueError, match="generate the word first"):
        await eng.generate(_entry(wid, sid, status="pending"))


async def test_grade_works_on_never_persisted_question(session_factory):
    wid, sid = await _seed_word(session_factory)
    eng = _engine(session_factory)
    q = (await eng.generate(_entry(wid, sid), formats=["definition_mcq"], n=1))[0]
    assert q.id is None  # never stored
    score = await eng.grade(q, q.payload["correct_index"])
    assert score.correct is True  # grading needs no DB


async def test_delete_removes_row_and_get_returns_none(session_factory):
    wid, sid = await _seed_word(session_factory)
    mcq = GeneratedMCQ(stem="s", correct="eloquent", distractors=["terse", "dull"])
    eng = _engine(session_factory, llm=_fake_llm(mcq))
    q = (await eng.generate(_entry(wid, sid), formats=["contextual_mcq"], n=1))[0]
    assert await eng.delete(q.id) is True
    assert await eng.get(q.id) is None
    assert await _count(session_factory) == 0


async def test_engine_is_blind_to_persistence(session_factory):
    """A dummy persisting plugin and a dummy ephemeral plugin share one engine path.

    The engine calls ``generate`` identically for both; persistence happens only
    because one plugin chose to call ``ctx.store``. The engine has no branch on it.
    """
    persisted_flag = {"stored": False}

    class _Persisting:
        format = "definition_mcq"  # reuse a valid id for the coupling guard
        answer_kind = "single_choice"

        async def generate(self, ctx, n=1):
            q = Question(
                id=None,
                word_id=ctx.entry.word_id,
                sense_id=None,
                format=self.format,
                answer_kind=self.answer_kind,
                payload={"stem": "s", "options": ["eloquent", "x"], "correct_index": 0},
            )
            stored = await ctx.store.insert(q)
            persisted_flag["stored"] = True
            return [stored]

        async def grade(self, ctx, question, answer):
            return await grade_single_choice(question, answer)

    class _Ephemeral(_Persisting):
        async def generate(self, ctx, n=1):
            payload = {"stem": "s", "options": ["eloquent", "x"], "correct_index": 0}
            return [
                Question(
                    id=None,
                    word_id=ctx.entry.word_id,
                    sense_id=None,
                    format=self.format,
                    answer_kind=self.answer_kind,
                    payload=payload,
                )
            ]

    wid, sid = await _seed_word(session_factory)
    entry = _entry(wid, sid)

    eng = _engine(session_factory)
    eng._plugins["definition_mcq"] = _Persisting()  # inject via the same dispatch slot
    qs = await eng.generate(entry, formats=["definition_mcq"], n=1)
    assert persisted_flag["stored"] and qs[0].id is not None and await _count(session_factory) == 1

    eng2 = _engine(session_factory)
    eng2._plugins["definition_mcq"] = _Ephemeral()
    qs2 = await eng2.generate(entry, formats=["definition_mcq"], n=1)
    assert qs2[0].id is None and await _count(session_factory) == 1  # still 1, ephemeral added none


# --- 5. registry / extensibility ------------------------------------------


def test_registry_has_seed_formats():
    assert set(REGISTRY) == {
        "definition_mcq",
        "cloze",
        "contextual_mcq",
        "use_in_sentence",
        "matching",
        "listening",
        "spelling",
        "pronunciation_mcq",
        "collocation_fill",
    }
    assert REGISTRY["definition_mcq"].answer_kind == "single_choice"
    assert REGISTRY["cloze"].answer_kind == "text_span"
    assert REGISTRY["use_in_sentence"].answer_kind == "free_text"
    assert REGISTRY["matching"].answer_kind == "matching"
    assert REGISTRY["listening"].answer_kind == "single_choice"
    assert REGISTRY["spelling"].answer_kind == "text_span"
    assert REGISTRY["pronunciation_mcq"].answer_kind == "single_choice"
    assert REGISTRY["collocation_fill"].answer_kind == "text_span"


def test_register_rejects_out_of_vocab_format():
    with pytest.raises(ValueError, match="unknown question format"):
        register(FormatSpec("not_a_format", "single_choice", DefinitionMCQ))


def test_register_rejects_coupling_drift():
    class _Mislabeled:
        format = "cloze"
        answer_kind = "free_text"  # disagrees with the spec below

        async def generate(self, ctx, n=1):
            return []

        async def grade(self, ctx, question, answer):
            return None

    with pytest.raises(ValueError, match="answer_kind"):
        register(FormatSpec("cloze", "single_choice", _Mislabeled))


async def test_one_line_extensibility_new_plugin(session_factory):
    """Register a brand-new format with ONE register() call; generate+grade it via
    the engine with no core-file edit (the plugin lives entirely in this test)."""
    from lexi_ai import constants
    from lexi_ai.questions import base

    class DummyPlugin:
        format = "dummy_reverse"
        answer_kind = "single_choice"

        async def generate(self, ctx, n=1):
            return [
                Question(
                    id=None,
                    word_id=ctx.entry.word_id,
                    sense_id=None,
                    format=self.format,
                    answer_kind=self.answer_kind,
                    payload={
                        "stem": "pick eloquent",
                        "options": ["eloquent", "x"],
                        "correct_index": 0,
                    },
                )
            ]

        async def grade(self, ctx, question, answer):
            return await grade_single_choice(question, answer)

    # Extend the vocab + registry the way a real new format would (test-local).
    original_formats = constants.QUESTION_FORMATS
    constants.QUESTION_FORMATS = original_formats | {"dummy_reverse"}
    base.QUESTION_FORMATS = constants.QUESTION_FORMATS
    try:
        register(FormatSpec("dummy_reverse", "single_choice", DummyPlugin))
        wid, sid = await _seed_word(session_factory)
        eng = _engine(session_factory)
        qs = await eng.generate(_entry(wid, sid), formats=["dummy_reverse"], n=1)
        assert len(qs) == 1
        score = await eng.grade(qs[0], 0)
        assert score.correct is True
    finally:
        REGISTRY.pop("dummy_reverse", None)
        constants.QUESTION_FORMATS = original_formats
        base.QUESTION_FORMATS = original_formats


# --- 5b. matching format (rule, new answer_kind) --------------------------


def _multi_sense_entry(word_id, *, norm="set", guides_defs=None) -> Entry:
    """A done entry with N guideworded senses — the matching format's input."""
    guides_defs = guides_defs or [
        ("PUT", "to place something in a position"),
        ("READY", "to prepare or arrange for use"),
        ("GROUP", "a collection of similar things"),
    ]
    return Entry(
        display=norm,
        norm=norm,
        entry_type="word",
        pos="verb",
        status="done",
        word_id=word_id,
        senses=[
            SenseView(
                definition=d,
                tier="core",
                pos="verb",
                cefr_level=None,
                guideword=g,
                sense_id=i,
            )
            for i, (g, d) in enumerate(guides_defs, start=1)
        ],
    )


def test_matching_registered_with_own_answer_kind():
    assert REGISTRY["matching"].answer_kind == "matching"
    plugin = Matching()
    assert plugin.format == "matching" and plugin.answer_kind == "matching"


async def test_matching_pairs_guidewords_to_definitions(session_factory):
    wid, _sid = await _seed_word(session_factory)
    guides_defs = [
        ("PUT", "to place something in a position"),
        ("READY", "to prepare or arrange for use"),
        ("GROUP", "a collection of similar things"),
    ]
    entry = _multi_sense_entry(wid, guides_defs=guides_defs)
    qs = await Matching().generate(_mcq_ctx(entry), n=1)
    assert len(qs) == 1
    p = qs[0].payload
    assert qs[0].answer_kind == "matching"
    # Cue column is the guidewords in stable sense order.
    assert p["lefts"] == ["PUT", "READY", "GROUP"]
    # Definition column holds exactly the definitions (shuffled — a permutation).
    defs = [d for _g, d in guides_defs]
    assert sorted(p["rights"]) == sorted(defs)
    # correct_map is a permutation aligning each left to its OWN definition.
    assert sorted(p["correct_map"]) == [0, 1, 2]
    defs_by_guide = dict(guides_defs)
    for i, g in enumerate(p["lefts"]):
        assert p["rights"][p["correct_map"][i]] == defs_by_guide[g]


async def test_matching_single_sense_returns_empty(session_factory):
    wid, sid = await _seed_word(session_factory)
    # The default single-sense entry has nothing to match against -> [].
    qs = await Matching().generate(_mcq_ctx(_entry(wid, sid)), n=1)
    assert qs == []


async def test_matching_requires_guidewords(session_factory):
    wid, _sid = await _seed_word(session_factory)
    # Two senses but no guidewords -> no cue column -> [] (no fabricated cues).
    entry = _multi_sense_entry(wid, guides_defs=[("PUT", "def one"), ("READY", "def two")])
    for s in entry.senses:
        s.guideword = None
    qs = await Matching().generate(_mcq_ctx(entry), n=1)
    assert qs == []


async def test_matching_option_order_is_deterministic(session_factory):
    wid, _sid = await _seed_word(session_factory)
    entry = _multi_sense_entry(wid)
    a = await Matching().generate(_mcq_ctx(entry), n=1)
    b = await Matching().generate(_mcq_ctx(entry), n=1)
    assert a[0].payload["rights"] == b[0].payload["rights"]  # seeded shuffle, stable


def _matching_question(lefts, rights, correct_map) -> Question:
    return Question(
        id=None,
        word_id=1,
        sense_id=None,
        format="matching",
        answer_kind="matching",
        payload={"prompt": "Match each.", "lefts": lefts, "rights": rights,
                 "correct_map": correct_map},
    )


async def test_grade_matching_all_correct():
    q = _matching_question(["a", "b", "c"], ["A", "B", "C"], [0, 1, 2])
    score = await grade_matching(q, [0, 1, 2])
    assert score.correct is True and score.score == 1.0 and score.kind == "rule"


async def test_grade_matching_partial_credit():
    q = _matching_question(["a", "b", "c", "d"], ["A", "B", "C", "D"], [0, 1, 2, 3])
    # two of four right -> 0.5, not fully correct
    score = await grade_matching(q, [0, 1, 3, 2])
    assert score.correct is False and score.score == 0.5


async def test_grade_matching_order_independent_via_correct_map():
    # A shuffled right column is encoded in correct_map; submitting correct_map
    # back scores full regardless of display order (grading compares indices).
    q = _matching_question(["a", "b", "c"], ["C", "A", "B"], [1, 2, 0])
    score = await grade_matching(q, [1, 2, 0])
    assert score.correct is True and score.score == 1.0


async def test_grade_matching_wrong_length_is_wrong_not_error():
    q = _matching_question(["a", "b"], ["A", "B"], [0, 1])
    score = await grade_matching(q, [0])  # too short — scored, not raised
    assert score.correct is False and score.kind == "rule"


# --- 5b. audio formats: listening + spelling ------------------------------


def test_listening_and_spelling_registered():
    assert REGISTRY["listening"].answer_kind == "single_choice"
    assert REGISTRY["spelling"].answer_kind == "text_span"
    assert Listening().answer_kind == "single_choice"
    assert Spelling().answer_kind == "text_span"


async def test_listening_carries_audio_ref_tuple_not_row_id(session_factory):
    wid, sid = await _seed_word(session_factory)
    tts = FakeTtsPort()
    qs = await Listening().generate(_mcq_ctx(_entry(wid, sid), tts=tts), n=1)
    assert len(qs) == 1
    p = qs[0].payload
    assert qs[0].answer_kind == "single_choice"
    # The clip was requested for the core sense's definition audio.
    assert tts.calls == [("sense_def", sid)]
    # The payload stores the (source_kind, source_id, voice, fmt) REFERENCE tuple,
    # never a raw assets-row id (a row id dangles after a purge).
    ref = p["audio_ref"]
    assert ref == {"source_kind": "sense_def", "source_id": sid, "voice": "alloy", "fmt": "mp3"}
    # It is a single_choice MCQ: the answer word is among the options.
    assert p["options"][p["correct_index"]] == "eloquent"
    assert len(p["options"]) == 4


async def test_spelling_carries_audio_ref_and_answer_norm(session_factory):
    wid, sid = await _seed_word(session_factory)
    tts = FakeTtsPort()
    qs = await Spelling().generate(_mcq_ctx(_entry(wid, sid), tts=tts), n=1)
    assert len(qs) == 1
    p = qs[0].payload
    assert qs[0].answer_kind == "text_span"
    assert tts.calls == [("sense_def", sid)]
    assert p["audio_ref"] == {
        "source_kind": "sense_def",
        "source_id": sid,
        "voice": "alloy",
        "fmt": "mp3",
    }
    assert p["answer_norm"] == "eloquent"


async def test_listening_without_tts_is_unavailable(session_factory):
    wid, sid = await _seed_word(session_factory)
    # ctx.tts is None -> format unavailable -> [] (best-effort, no raise).
    qs = await Listening().generate(_mcq_ctx(_entry(wid, sid)), n=1)
    assert qs == []


async def test_spelling_without_tts_is_unavailable(session_factory):
    wid, sid = await _seed_word(session_factory)
    qs = await Spelling().generate(_mcq_ctx(_entry(wid, sid)), n=1)
    assert qs == []


async def test_audio_formats_degrade_when_clip_cannot_be_made(session_factory):
    wid, sid = await _seed_word(session_factory)
    # The port cannot make a clip (source text gone) -> no fabricated asset -> [].
    tts = FakeTtsPort(missing=True)
    assert await Listening().generate(_mcq_ctx(_entry(wid, sid), tts=tts), n=1) == []
    assert await Spelling().generate(_mcq_ctx(_entry(wid, sid), tts=tts), n=1) == []


async def test_listening_degrades_when_no_distractors(session_factory):
    wid, sid = await _seed_word(session_factory)
    ctx = QuestionContext(
        entry=_entry(wid, sid), distractors=FakeDistractors(pool=[]), tts=FakeTtsPort()
    )
    assert await Listening().generate(ctx, n=1) == []  # no options -> no question


async def test_listening_grades_by_index_shared_helper(session_factory):
    wid, sid = await _seed_word(session_factory)
    tts = FakeTtsPort()
    q = (await Listening().generate(_mcq_ctx(_entry(wid, sid), tts=tts), n=1))[0]
    ctx = QuestionContext(entry=None, distractors=FakeDistractors())
    score = await Listening().grade(ctx, q, q.payload["correct_index"])
    assert score.correct is True and score.kind == "rule"  # same rule helper as MCQ


async def test_spelling_grades_by_matchkey_shared_helper(session_factory):
    wid, sid = await _seed_word(session_factory)
    tts = FakeTtsPort()
    q = (await Spelling().generate(_mcq_ctx(_entry(wid, sid), tts=tts), n=1))[0]
    ctx = QuestionContext(entry=None, distractors=FakeDistractors())
    assert (await Spelling().grade(ctx, q, "ELOQUENT")).correct is True  # case folds
    assert (await Spelling().grade(ctx, q, "wrong")).correct is False


async def test_audio_grading_independent_of_ref_resolvability(session_factory):
    # A stored audio_ref that no longer resolves (clip purged / source regenerated)
    # must NOT block grading: grading is index/text based and never touches the clip.
    wid, sid = await _seed_word(session_factory)
    tts = FakeTtsPort()
    lq = (await Listening().generate(_mcq_ctx(_entry(wid, sid), tts=tts), n=1))[0]
    sq = (await Spelling().generate(_mcq_ctx(_entry(wid, sid), tts=tts), n=1))[0]
    ctx = QuestionContext(entry=None, distractors=FakeDistractors())  # no tts at grade time
    assert (await Listening().grade(ctx, lq, lq.payload["correct_index"])).correct is True
    assert (await Spelling().grade(ctx, sq, "eloquent")).correct is True


# --- 6. payload round-trip edges ------------------------------------------


async def test_payload_unicode_round_trips(session_factory):
    wid, _sid = await _seed_word(session_factory)
    repo = QuestionRepository(session_factory)
    payload = {"stem_with_blank": "café ____ naïve", "answer_norm": "résumé", "opts": ["a", "b"]}
    q = await repo.insert(
        Question(
            id=None,
            word_id=wid,
            sense_id=None,
            format="cloze",
            answer_kind="text_span",
            payload=payload,
        )
    )
    got = await repo.get(q.id)
    assert got.payload == payload  # nested + unicode survive byte-for-byte


async def test_payload_nul_is_rejected(session_factory):
    wid, _sid = await _seed_word(session_factory)
    repo = QuestionRepository(session_factory)
    with pytest.raises(ValueError, match="NUL"):
        await repo.insert(
            Question(
                id=None,
                word_id=wid,
                sense_id=None,
                format="cloze",
                answer_kind="text_span",
                payload={"stem_with_blank": "a\x00b", "answer_norm": "x"},
            )
        )


# --- pronunciation_mcq + collocation_fill (reuse stored IPA / collocations) ---


async def test_pronunciation_mcq_builds_from_ipa(session_factory):
    from lexi_ai.questions.formats import PronunciationMCQ

    wid, sid = await _seed_word(session_factory)
    entry = _entry(wid, sid, ipa_uk="ˈɛl.ə.kwənt")
    qs = await PronunciationMCQ().generate(_mcq_ctx(entry), n=1)
    assert len(qs) == 1
    payload = qs[0].payload
    assert "ɛl" in payload["stem"]  # IPA appears in the stem
    assert payload["options"][payload["correct_index"]] == "eloquent"
    assert payload["options"].count("eloquent") == 1  # answer not duplicated


async def test_pronunciation_mcq_without_ipa_is_unavailable(session_factory):
    wid, sid = await _seed_word(session_factory)
    qs = await PronunciationMCQ().generate(_mcq_ctx(_entry(wid, sid)), n=1)  # no IPA
    assert qs == []


async def test_pronunciation_mcq_falls_back_to_us_ipa(session_factory):
    from lexi_ai.questions.formats import PronunciationMCQ

    wid, sid = await _seed_word(session_factory)
    entry = _entry(wid, sid, ipa_us="ˈɛl.ə.kwənt")  # only US IPA present
    qs = await PronunciationMCQ().generate(_mcq_ctx(entry), n=1)
    assert len(qs) == 1


async def test_collocation_fill_blanks_target(session_factory):

    wid, sid = await _seed_word(session_factory, norm="rain", definition="water from clouds")
    entry = _entry(wid, sid, norm="rain", collocations=["heavy rain", "acid rain"])
    qs = await CollocationFill().generate(_mcq_ctx(entry), n=1)
    assert len(qs) == 1
    stem = qs[0].payload["stem_with_blank"]
    assert "_____" in stem and "rain" not in stem.lower()
    assert qs[0].payload["answer_norm"] == "rain"


async def test_collocation_fill_folds_inflected_surface(session_factory):
    # "heavy rains" blanks against the lemma "rain" via the accepted-forms set.

    wid, sid = await _seed_word(session_factory, norm="rain", definition="water from clouds")
    entry = _entry(
        wid, sid, norm="rain", collocations=["heavy rains"], forms=[FormView("plural", "rains")]
    )
    qs = await CollocationFill().generate(_mcq_ctx(entry), n=1)
    assert len(qs) == 1
    assert "_____" in qs[0].payload["stem_with_blank"]
    assert "rains" in qs[0].payload["accepted_forms"]


async def test_collocation_fill_without_collocations_is_unavailable(session_factory):

    wid, sid = await _seed_word(session_factory)
    qs = await CollocationFill().generate(_mcq_ctx(_entry(wid, sid)), n=1)  # no collocations
    assert qs == []


def test_new_formats_registered():
    assert "pronunciation_mcq" in REGISTRY
    assert "collocation_fill" in REGISTRY
