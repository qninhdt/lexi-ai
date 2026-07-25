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
    GeneratedForm,
    GeneratedResult,
    GeneratedSense,
    GeneratedSenseRelation,
    RelatedWord,
)
from lexi_ai.infrastructure.db.models import (
    Collocation,
    SenseForm,
    SenseRelation,
    Word,
    WordRelation,
)
from lexi_ai.normalize import match_key
from tests.support.persistence_driver import PersistenceDriver

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
    ipa_uk: str | None = None,
    ipa_us: str | None = None,
    forms: list[tuple[str, str]] | None = None,
    domain: str | None = None,
    usage_note: str | None = None,
    sense_relations: list[tuple[str, str, str]] | None = None,
) -> GeneratedResult:
    """One-unit result whose single sense carries the given enrichments.

    ``related`` is a list of ``(norm, rel_type)`` pairs -> word-references.
    ``forms`` is a list of ``(inf, surface)`` pairs -> inflection paradigm.
    ``sense_relations`` is a list of ``(rel_type, norm, gloss)`` triples ->
    sense-level half-edges emitted by the single sense.
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
                        pos="noun",
                        guideword=guideword,
                        grammar=grammar or [],
                        register=register,
                        connotation=connotation,
                        collocations=collocations or [],
                        ipa_uk=ipa_uk,
                        ipa_us=ipa_us,
                        forms=[GeneratedForm(inf=i, surface=s) for i, s in (forms or [])],
                        domain=domain,
                        usage_note=usage_note,
                        relations=[
                            GeneratedSenseRelation(rel_type=rt, norm=n, gloss=g)
                            for rt, n, g in (sense_relations or [])
                        ],
                    )
                ],
                related=[RelatedWord(norm=n, rel_type=rt) for n, rt in (related or [])],
            )
        ]
    )


async def _reading_lexicon(engine, repo) -> Lexicon:
    # Reads only — no generator/loader needed (mirrors test_tags.py::_seed_browse).
    return Lexicon(create_session_factory(engine), None, None, engine=engine)  # type: ignore[arg-type]


# --- word-references: normalization + dedup (the user's critical ask) ------


async def test_word_family_normalized_and_shared(engine):
    # "happy" names word_family "Happiness" (capitalized) -> a words row keyed by
    # match_key("happiness"); a second word referencing "happiness" shares it.
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
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
                    select(WordRelation).where(
                        WordRelation.to_word_id == target_id,
                        WordRelation.rel_type == "word_family",
                    )
                )
            ).scalars()
        )
        assert len(links) == 2


async def test_confused_with_links(engine):
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    words = await repo.persist_result(_result("affect", related=[("effect", "confused_with")]))
    lex = await _reading_lexicon(engine, repo)
    entry = await lex.get_entry(words[0].id)

    confused = [ln for ln in entry.links if ln.rel_type == "confused_with"]
    assert len(confused) == 1
    assert confused[0].display == "effect"


async def test_hypernym_hyponym_sense_relations(engine):
    # Two taxonomic relations are now SENSE-level (Phase 3): the emitting sense
    # carries them with a gloss, and they persist as sense_relation half-edges
    # (to_sense_id NULL = pending) keyed to real stub words rows.
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    words = await repo.persist_result(
        _result(
            "dog",
            sense_relations=[
                ("hypernym", "animal", "a living creature"),
                ("hyponym", "poodle", "a curly-haired dog breed"),
            ],
        )
    )

    async with sf() as s:
        sense_id = (await s.execute(select(SenseRelation.from_sense_id).limit(1))).scalar_one()
        rels = list(
            (
                await s.execute(
                    select(SenseRelation).where(SenseRelation.from_sense_id == sense_id)
                )
            ).scalars()
        )
        by_type = {r.rel_type: r for r in rels}
        assert set(by_type) == {"hypernym", "hyponym"}
        # Half-edges: target word resolved, target sense still pending (NULL).
        assert by_type["hypernym"].to_word_id is not None
        assert by_type["hypernym"].to_sense_id is None
        assert by_type["hypernym"].resolve_attempted_at is None
        assert by_type["hypernym"].gloss == "a living creature"
        assert by_type["hyponym"].gloss == "a curly-haired dog breed"

    # Word-level links list stays empty — these are NOT word relations anymore.
    lex = await _reading_lexicon(engine, repo)
    entry = await lex.get_entry(words[0].id)
    assert all(ln.rel_type not in ("hypernym", "hyponym") for ln in entry.links)


async def test_meronym_holonym_sense_relations(engine):
    # Part-whole relations are also SENSE-level; same half-edge persistence path.
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    await repo.persist_result(
        _result(
            "car",
            sense_relations=[
                ("meronym", "wheel", "round part a car rolls on"),
                ("holonym", "vehicle", "a machine for transport"),
            ],
        )
    )

    async with sf() as s:
        rels = list((await s.execute(select(SenseRelation))).scalars())
        by_type = {r.rel_type: r for r in rels}
        assert set(by_type) == {"meronym", "holonym"}
        assert by_type["meronym"].to_word_id is not None
        assert by_type["meronym"].to_sense_id is None
        assert by_type["holonym"].to_word_id is not None


async def test_sense_relation_empty_gloss_skipped(engine):
    # [F12] gloss is the load-bearing WSD signal — a blank one (after sanitize)
    # gets the whole half-edge dropped, never persisted as a dead row.
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    await repo.persist_result(
        _result(
            "bright",
            sense_relations=[
                ("antonym", "dark", "   \x00  "),  # whitespace/ctrl-only -> empty
                ("synonym", "brilliant", "shining vividly"),
            ],
        )
    )
    async with sf() as s:
        rels = list((await s.execute(select(SenseRelation))).scalars())
        assert {r.rel_type for r in rels} == {"synonym"}  # antonym dropped


async def test_sense_relation_self_reference_skipped(engine):
    # [Case 8] a sense-level relation whose target normalizes to the emitting
    # sense's OWN word is vacuous and never persisted.
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    await repo.persist_result(
        _result("Happy", sense_relations=[("synonym", "happy", "feeling joy")])
    )
    async with sf() as s:
        count = (await s.execute(select(func.count()).select_from(SenseRelation))).scalar_one()
        assert count == 0


async def test_sense_relation_dedup_on_triple(engine):
    # Dedup mirrors _ensure_link: the UNIQUE (from_sense, to_word, rel_type)
    # triple collapses duplicate emissions to one row (last gloss not overwritten).
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    await repo.persist_result(
        _result(
            "big",
            sense_relations=[
                ("synonym", "large", "of great size"),
                ("synonym", "large", "sizeable"),  # same triple -> deduped
            ],
        )
    )
    async with sf() as s:
        count = (await s.execute(select(func.count()).select_from(SenseRelation))).scalar_one()
        assert count == 1


# --- sense labels: round-trip + lossless grammar join/split ----------------


async def test_sense_labels_round_trip(engine):
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
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
    entry = await lex.get_entry(words[0].id)
    sense = entry.senses[0]

    assert sense.guideword == "PUT"
    assert sense.register == "formal"
    assert sense.connotation == "neutral"
    # Multi-token grammar joins on the write path and splits back losslessly.
    assert sense.grammar == ["transitive", "verb + to-infinitive"]


async def test_ipa_surfaces_on_sense_view(engine):
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    words = await repo.persist_result(_result("lead", ipa_uk="liːd", ipa_us="liːd"))
    lex = await _reading_lexicon(engine, repo)
    entry = await lex.get_entry(words[0].id)
    sense = entry.senses[0]

    assert sense.ipa_uk == "liːd"
    assert sense.ipa_us == "liːd"


# --- enum rejection at schema validation (closed vocab) --------------------


def test_grammar_labels_have_no_comma():
    # grammar is stored comma-joined in ONE column and split back on read. A label
    # containing ',' would silently corrupt that round-trip (bogus split tokens),
    # so the separator must never appear in the vocabulary. Locks the invariant.
    assert all("," not in label for label in GRAMMAR_LABELS)


def test_grammar_rejects_out_of_vocab():
    with pytest.raises(ValueError):
        GeneratedSense(definition="d", tier="core", pos="noun", grammar=["bogus"])


def test_register_connotation_reject_out_of_vocab():
    with pytest.raises(ValueError):
        GeneratedSense(definition="d", tier="core", pos="noun", register="sarcastic")
    with pytest.raises(ValueError):
        GeneratedSense(definition="d", tier="core", pos="noun", connotation="mixed")


# --- collocations: ordering + sanitization + best-effort skip --------------


async def test_collocations_ordered_and_sanitized(engine):
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    words = await repo.persist_result(
        _result(
            "rain",
            collocations=["heavy rain", "rain\x00storm", "pouring\nrain", "   ", "light rain"],
        )
    )
    lex = await _reading_lexicon(engine, repo)
    entry = await lex.get_entry(words[0].id)
    cols = entry.senses[0].collocations

    # NUL/newline collapsed to a space, whitespace-only dropped, order preserved.
    assert cols == ["heavy rain", "rain storm", "pouring rain", "light rain"]
    assert all("\x00" not in c and "\n" not in c for c in cols)


async def test_collocations_empty_string_skipped(engine):
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    words = await repo.persist_result(_result("void", collocations=["", "real phrase", ""]))
    async with create_session_factory(engine)() as s:
        rows = list((await s.execute(select(Collocation))).scalars())
    assert [r.text for r in rows] == ["real phrase"]
    assert words[0].status == "done"


# --- best-effort: an enrichment-free sense still persists done -------------


async def test_empty_enrichment_persists_done(engine):
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    words = await repo.persist_result(_result("plain"))
    assert words[0].status == "done"

    lex = await _reading_lexicon(engine, repo)
    entry = await lex.get_entry(words[0].id)
    sense = entry.senses[0]
    assert sense.guideword is None
    assert sense.grammar == []
    assert sense.register is None
    assert sense.connotation is None
    assert sense.collocations == []
    assert sense.forms == []
    assert sense.domain is None
    assert sense.usage_note is None


# --- eager-load: collocations survive session close ------------------------


async def test_collocations_no_detached_load(engine):
    # get() returns a detached Entry (session closed in _to_entry). Accessing
    # collocations must NOT raise — it is selectinload-ed, not lazy.
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    words = await repo.persist_result(_result("draw", collocations=["draw a line", "draw water"]))
    lex = await _reading_lexicon(engine, repo)
    entry = await lex.get_entry(words[0].id)

    # No DetachedInstanceError / MissingGreenlet on this access.
    assert entry.senses[0].collocations == ["draw a line", "draw water"]


# --- write-path sanity: grammar stored comma-joined, empty -> None ---------


async def test_grammar_stored_joined_empty_is_none(engine):
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    await repo.persist_result(_result("go", grammar=["intransitive"]))
    await repo.persist_result(_result("stay"))  # no grammar

    from lexi_ai.infrastructure.db.models import Sense

    async with create_session_factory(engine)() as s:
        go = (await s.execute(select(Sense).join(Word).where(Word.norm == "go"))).scalar_one()
        stay = (await s.execute(select(Sense).join(Word).where(Word.norm == "stay"))).scalar_one()
    assert go.grammar == "intransitive"
    assert stay.grammar is None  # empty list -> None, not "" (clean [] on read)


async def test_word_family_count_and_no_self_link(engine):
    # A word referencing ITSELF as a relative must not create a self-link.
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    words = await repo.persist_result(_result("happy", related=[("happy", "word_family")]))
    async with create_session_factory(engine)() as s:
        n = (
            await s.execute(
                select(func.count(WordRelation.id)).where(WordRelation.from_word_id == words[0].id)
            )
        ).scalar()
    assert n == 0  # self-reference skipped


# --- inflection forms: paradigm round-trip + ordering + sanitization --------


async def test_forms_round_trip_ordered(engine):
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    words = await repo.persist_result(
        _result(
            "run",
            forms=[
                ("base", "run"),
                ("past", "ran"),
                ("past_participle", "run"),
                ("present_3sg", "runs"),
                ("ing", "running"),
            ],
        )
    )
    lex = await _reading_lexicon(engine, repo)
    entry = await lex.get_entry(words[0].id)
    forms = entry.senses[0].forms

    # Full paradigm surfaces in emit order, each (inf, surface) preserved.
    assert [(f.inf, f.surface) for f in forms] == [
        ("base", "run"),
        ("past", "ran"),
        ("past_participle", "run"),
        ("present_3sg", "runs"),
        ("ing", "running"),
    ]


async def test_forms_sanitized_and_empty_skipped(engine):
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    words = await repo.persist_result(
        _result(
            "dream",
            forms=[("past", "dreamed"), ("past", "dream\x00t"), ("ing", "   ")],
        )
    )
    async with create_session_factory(engine)() as s:
        rows = list((await s.execute(select(SenseForm).order_by(SenseForm.form_order))).scalars())
    # NUL collapsed to space; whitespace-only surface dropped; a repeated label
    # (dreamed/dreamt) is allowed — forms is a flat list, not a dict.
    assert [(r.inf, r.surface) for r in rows] == [("past", "dreamed"), ("past", "dream t")]
    assert all("\x00" not in r.surface for r in rows)
    assert words[0].status == "done"


def test_forms_inf_rejects_out_of_vocab():
    with pytest.raises(ValueError):
        GeneratedForm(inf="gerundive", surface="x")


# --- domain + usage_note: free-text round-trip + sanitization --------------


async def test_domain_and_usage_note_round_trip(engine):
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    words = await repo.persist_result(
        _result(
            "cache",
            domain="computing",
            usage_note="In tech, distinct from 'cash' (money).",
        )
    )
    lex = await _reading_lexicon(engine, repo)
    sense = (await lex.get_entry(words[0].id)).senses[0]

    assert sense.domain == "computing"
    assert sense.usage_note == "In tech, distinct from 'cash' (money)."


async def test_domain_usage_note_sanitized_and_empty_is_none(engine):
    sf = create_session_factory(engine)
    repo = PersistenceDriver(sf)
    words = await repo.persist_result(
        _result("mixed", domain="med\x00icine", usage_note="line\none")
    )
    plain = await repo.persist_result(_result("bare"))  # neither field
    lex = await _reading_lexicon(engine, repo)

    marked = (await lex.get_entry(words[0].id)).senses[0]
    assert marked.domain == "med icine"  # NUL collapsed to space
    assert marked.usage_note == "line one"  # newline collapsed

    bare = (await lex.get_entry(plain[0].id)).senses[0]
    assert bare.domain is None and bare.usage_note is None
