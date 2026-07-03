"""Tests for word-content enrichment (word-refs + sense labels + collocations).

Locks the write->read spine with a fake generator (no network): word-reference
normalization/dedup (word_family/confused_with ride the related[] path), enum
rejection at schema validation (grammar/register/connotation), sense-label and
collocation round-trip through ``SenseView``, control-char sanitization, and
best-effort empties. In-memory SQLite + StaticPool, mirroring ``test_tags.py``.
"""

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from lexi_ai.api import Lexicon
from lexi_ai.constants import GRAMMAR_LABELS
from lexi_ai.db import create_session_factory, init_models
from lexi_ai.generation.schemas import (
    GeneratedEntry,
    GeneratedResult,
    GeneratedSense,
    RelatedWord,
)
from lexi_ai.models import Collocation, EntryLink, Word
from lexi_ai.normalize import match_key
from lexi_ai.persistence.repository import Repository

# --- write-path harness ----------------------------------------------------


@pytest.fixture
async def engine():
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
    yield engine
    await engine.dispose()


def _result(
    norm: str,
    *,
    related: list[tuple[str, str]] | None = None,
    guideword: str | None = None,
    grammar: list[str] | None = None,
    register: str | None = None,
    connotation: str | None = None,
    collocations: list[str] | None = None,
) -> GeneratedResult:
    """One-unit result whose single sense carries the given enrichments.

    ``related`` is a list of ``(norm, rel_type)`` pairs -> word-references.
    """
    return GeneratedResult(
        units=[
            GeneratedEntry(
                norm=norm,
                entry_type="word",
                senses=[
                    GeneratedSense(
                        definition=f"def of {norm}",
                        tier="core",
                        guideword=guideword,
                        grammar=grammar or [],
                        register=register,
                        connotation=connotation,
                        collocations=collocations or [],
                    )
                ],
                related=[RelatedWord(norm=n, rel_type=rt) for n, rt in (related or [])],
            )
        ]
    )


async def _reading_lexicon(engine, repo) -> Lexicon:
    # Reads only — no generator/loader needed (mirrors test_tags.py::_seed_browse).
    return Lexicon(create_session_factory(engine), None, None, repo, engine=engine)  # type: ignore[arg-type]


# --- word-references: normalization + dedup (the user's critical ask) ------


async def test_word_family_normalized_and_shared(engine):
    # "happy" names word_family "Happiness" (capitalized) -> a words row keyed by
    # match_key("happiness"); a second word referencing "happiness" shares it.
    sf = create_session_factory(engine)
    repo = Repository(sf)
    await repo.persist_result(_result("happy", related=[("Happiness", "word_family")]))
    await repo.persist_result(_result("cheerful", related=[("happiness", "word_family")]))

    async with sf() as s:
        key = match_key("happiness")
        rows = list((await s.execute(select(Word).where(Word.match_key == key))).scalars())
        assert len(rows) == 1  # Happiness / happiness fold to ONE normalized row
        target_id = rows[0].id
        # Both source words link to that one shared row.
        links = list(
            (
                await s.execute(
                    select(EntryLink).where(
                        EntryLink.to_word_id == target_id,
                        EntryLink.rel_type == "word_family",
                    )
                )
            ).scalars()
        )
        assert len(links) == 2


async def test_confused_with_links(engine):
    sf = create_session_factory(engine)
    repo = Repository(sf)
    words = await repo.persist_result(_result("affect", related=[("effect", "confused_with")]))
    lex = await _reading_lexicon(engine, repo)
    entry = await lex.get(words[0].id)

    confused = [ln for ln in entry.links if ln.rel_type == "confused_with"]
    assert len(confused) == 1
    assert confused[0].display == "effect"


# --- sense labels: round-trip + lossless grammar join/split ----------------


async def test_sense_labels_round_trip(engine):
    sf = create_session_factory(engine)
    repo = Repository(sf)
    words = await repo.persist_result(
        _result(
            "set",
            guideword="PUT",
            grammar=["transitive", "verb + to-infinitive"],
            register="formal",
            connotation="neutral",
        )
    )
    lex = await _reading_lexicon(engine, repo)
    entry = await lex.get(words[0].id)
    sense = entry.senses[0]

    assert sense.guideword == "PUT"
    assert sense.register == "formal"
    assert sense.connotation == "neutral"
    # Multi-token grammar joins on the write path and splits back losslessly.
    assert sense.grammar == ["transitive", "verb + to-infinitive"]


# --- enum rejection at schema validation (closed vocab) --------------------


def test_grammar_labels_have_no_comma():
    # grammar is stored comma-joined in ONE column and split back on read. A label
    # containing ',' would silently corrupt that round-trip (bogus split tokens),
    # so the separator must never appear in the vocabulary. Locks the invariant.
    assert all("," not in label for label in GRAMMAR_LABELS)


def test_grammar_rejects_out_of_vocab():
    with pytest.raises(ValueError):
        GeneratedSense(definition="d", tier="core", grammar=["bogus"])


def test_register_connotation_reject_out_of_vocab():
    with pytest.raises(ValueError):
        GeneratedSense(definition="d", tier="core", register="sarcastic")
    with pytest.raises(ValueError):
        GeneratedSense(definition="d", tier="core", connotation="mixed")


# --- collocations: ordering + sanitization + best-effort skip --------------


async def test_collocations_ordered_and_sanitized(engine):
    sf = create_session_factory(engine)
    repo = Repository(sf)
    words = await repo.persist_result(
        _result(
            "rain",
            collocations=["heavy rain", "rain\x00storm", "pouring\nrain", "   ", "light rain"],
        )
    )
    lex = await _reading_lexicon(engine, repo)
    entry = await lex.get(words[0].id)
    cols = entry.senses[0].collocations

    # NUL/newline collapsed to a space, whitespace-only dropped, order preserved.
    assert cols == ["heavy rain", "rain storm", "pouring rain", "light rain"]
    assert all("\x00" not in c and "\n" not in c for c in cols)


async def test_collocations_empty_string_skipped(engine):
    sf = create_session_factory(engine)
    repo = Repository(sf)
    words = await repo.persist_result(_result("void", collocations=["", "real phrase", ""]))
    async with create_session_factory(engine)() as s:
        rows = list((await s.execute(select(Collocation))).scalars())
    assert [r.text for r in rows] == ["real phrase"]
    assert words[0].status == "done"


# --- best-effort: an enrichment-free sense still persists done -------------


async def test_empty_enrichment_persists_done(engine):
    sf = create_session_factory(engine)
    repo = Repository(sf)
    words = await repo.persist_result(_result("plain"))
    assert words[0].status == "done"

    lex = await _reading_lexicon(engine, repo)
    entry = await lex.get(words[0].id)
    sense = entry.senses[0]
    assert sense.guideword is None
    assert sense.grammar == []
    assert sense.register is None
    assert sense.connotation is None
    assert sense.collocations == []


# --- eager-load: collocations survive session close ------------------------


async def test_collocations_no_detached_load(engine):
    # get() returns a detached Entry (session closed in _to_entry). Accessing
    # collocations must NOT raise — it is selectinload-ed, not lazy.
    sf = create_session_factory(engine)
    repo = Repository(sf)
    words = await repo.persist_result(_result("draw", collocations=["draw a line", "draw water"]))
    lex = await _reading_lexicon(engine, repo)
    entry = await lex.get(words[0].id)

    # No DetachedInstanceError / MissingGreenlet on this access.
    assert entry.senses[0].collocations == ["draw a line", "draw water"]


# --- write-path sanity: grammar stored comma-joined, empty -> None ---------


async def test_grammar_stored_joined_empty_is_none(engine):
    sf = create_session_factory(engine)
    repo = Repository(sf)
    await repo.persist_result(_result("go", grammar=["intransitive"]))
    await repo.persist_result(_result("stay"))  # no grammar

    from lexi_ai.models import Sense

    async with create_session_factory(engine)() as s:
        go = (await s.execute(select(Sense).join(Word).where(Word.norm == "go"))).scalar_one()
        stay = (await s.execute(select(Sense).join(Word).where(Word.norm == "stay"))).scalar_one()
    assert go.grammar == "intransitive"
    assert stay.grammar is None  # empty list -> None, not "" (clean [] on read)


async def test_word_family_count_and_no_self_link(engine):
    # A word referencing ITSELF as a relative must not create a self-link.
    sf = create_session_factory(engine)
    repo = Repository(sf)
    words = await repo.persist_result(_result("happy", related=[("happy", "word_family")]))
    async with create_session_factory(engine)() as s:
        n = (
            await s.execute(
                select(func.count(EntryLink.id)).where(EntryLink.from_word_id == words[0].id)
            )
        ).scalar()
    assert n == 0  # self-reference skipped
