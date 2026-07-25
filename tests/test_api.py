"""Tests for the search-centric lookup API.

In-memory SQLite + a fake generator (counts calls, returns canned results) and a
fake reference stack (no real Cambridge/WordNet I/O). Asserts the contract:
search→generate, generate is cache-first (0 LLM on a hit), homographs converge on
one entry, search flags generated words (no re-offer), get/status work by lexi id,
custom strings generate, and force regenerates.
"""

import asyncio

import pytest
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from lexi_ai.api import Lexicon
from lexi_ai.db import create_session_factory, init_models, session_scope
from lexi_ai.domain.models import VectorRecord
from lexi_ai.embeddings import Embedder
from lexi_ai.facades import LexiconEngine, LexiconReader
from lexi_ai.generation.schemas import (
    ExampleBatch,
    GeneratedEntry,
    GeneratedResult,
    RelatedWord,
)
from lexi_ai.infrastructure.db.models import Example, Sense, Word
from lexi_ai.infrastructure.vectors.memory_index import InMemoryVectorIndex
from lexi_ai.markup import parse_marked_example
from lexi_ai.normalize import match_key
from lexi_ai.read_models import Entry, SenseView
from lexi_ai.references.cambridge import CamRef
from lexi_ai.references.loader import ReferenceBundle


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


class FakeCambridge:
    """Minimal stand-in for CambridgeSource.

    ``words`` maps a Cambridge word_id -> (display, entry_type, [surface_forms]).
    ``resolve_exact`` matches any surface form's match_key; ``rank_similar`` is a
    no-op (fuzzy is covered by the real-data reference tests).
    """

    def __init__(self, words: dict[int, tuple[str, str, list[str]]]):
        self._words = words

    async def resolve_exact(self, raw: str) -> list[CamRef]:
        key = match_key(raw)
        out = []
        for wid, (display, etype, surfaces) in self._words.items():
            if any(match_key(s) == key for s in surfaces):
                out.append(CamRef(word_id=wid, display_form=display, entry_type=etype))
        return out

    async def rank_similar(self, raw: str, limit: int = 10) -> list[tuple[CamRef, float]]:
        return []

    async def first_definitions(self, word_ids) -> dict[int, str]:
        return {wid: f"gloss of {self._words[wid][0]}" for wid in word_ids if wid in self._words}


class FakeLoader:
    """Fake reference stack. Bundles are keyed by the Cambridge word_id so a
    generated unit's norm can be canned per word."""

    def __init__(self, cambridge: FakeCambridge, norm_by_id: dict[int, str]):
        self._cambridge = cambridge
        self._norm_by_id = norm_by_id

    @property
    def cambridge(self) -> FakeCambridge:
        return self._cambridge

    async def bundle_by_id(self, cambridge_word_id: int) -> ReferenceBundle | None:
        norm = self._norm_by_id.get(cambridge_word_id)
        if norm is None:
            return None
        return ReferenceBundle(
            word_raw=norm, entry_type="word", cambridge_word_id=cambridge_word_id
        )

    async def bundle_custom(self, word: str) -> ReferenceBundle:
        return ReferenceBundle(word_raw=word, entry_type="word", cambridge_word_id=None)


class FakeGenerator:
    """Returns a canned GeneratedResult keyed by the bundle word_raw; counts calls."""

    def __init__(
        self,
        results_by_word: dict[str, GeneratedResult],
        example_batch: ExampleBatch | None = None,
    ):
        self._results = results_by_word
        self.calls = 0
        self.last_existing_tags: tuple[tuple[str, str], ...] = ()
        # Canned targeted-example output + recorders for the add_examples path.
        self._example_batch = example_batch
        self.example_calls = 0
        self.last_existing: list[str] | None = None
        self.last_n: int | None = None

    async def generate(self, bundle: ReferenceBundle, existing_tags=()) -> GeneratedResult:
        self.calls += 1
        self.last_existing_tags = tuple(existing_tags)
        await asyncio.sleep(0.01)  # let concurrent tasks overlap
        return self._results[match_key(bundle.word_raw)]

    async def generate_examples(self, sense, existing, n: int) -> ExampleBatch:
        self.example_calls += 1
        self.last_existing = list(existing)
        self.last_n = n
        return self._example_batch or ExampleBatch(examples=[])


def _entry(norm, aliases=None, related=None) -> GeneratedEntry:
    return GeneratedEntry(
        norm=norm,
        entry_type="word",
        pos="noun",
        senses=[
            {"definition": f"def of {norm}", "tier": "core", "cefr_level": "A1", "pos": "noun"}
        ],
        aliases=aliases or [],
        related=related or [],
    )


def _make_lexicon(
    engine,
    cam_words,
    norm_by_id,
    results_by_word,
    embedder=None,
    example_batch=None,
    vectors=None,
):
    session_factory = create_session_factory(engine)
    cambridge = FakeCambridge(cam_words)
    loader = FakeLoader(cambridge, norm_by_id)
    generator = FakeGenerator(results_by_word, example_batch=example_batch)
    lex = Lexicon(
        session_factory,
        loader,
        generator,
        engine=engine,
        embedder=embedder,
        vectors=vectors or InMemoryVectorIndex(),
    )
    return lex, generator, session_factory


async def test_search_then_generate_then_hit(engine):
    lex, gen, _sf = _make_lexicon(
        engine,
        cam_words={1: ("color", "word", ["color"])},
        norm_by_id={1: "color"},
        results_by_word={match_key("color"): GeneratedResult(units=[_entry("color")])},
    )
    results = await lex.reader().search("color")
    assert len(results) == 1
    assert not results[0].generated
    assert results[0].cambridge_id == 1

    entry = await lex.engine().generate(results[0])
    assert isinstance(entry, Entry)
    assert entry.display == "color"
    assert gen.calls == 1

    # Search again: the word is now generated (flagged), not re-offered.
    again = await lex.reader().search("color")
    assert again[0].generated
    assert again[0].lexi_word_id is not None
    # Generating the generated hit is a no-op (no LLM).
    entry2 = await lex.engine().generate(again[0])
    assert entry2.norm == "color"
    assert gen.calls == 1


async def test_homographs_converge_on_one_entry(engine):
    # Two Cambridge ids share one display (the C1 case). Before generation search
    # shows both (they are genuinely distinct Cambridge entries); generating either
    # converges on ONE row, and the other then folds into that generated word.
    lex, gen, session_factory = _make_lexicon(
        engine,
        cam_words={
            90264: ("shame on you", "phrase", ["shame on you"]),
            90265: ("shame on you", "idiom", ["shame on you"]),
        },
        norm_by_id={90264: "shame on you", 90265: "shame on you"},
        results_by_word={
            match_key("shame on you"): GeneratedResult(units=[_entry("shame on you")])
        },
    )
    results = await lex.reader().search("shame on you")
    exact = [r for r in results if r.score == 1.0]
    assert len(exact) == 2  # two distinct Cambridge entries, not yet generated

    await lex.engine().generate(exact[0])
    assert gen.calls == 1
    # The other homograph is now a cache hit (its word already exists), and the
    # display folds to a single generated result — no second row, no re-generation.
    after = await lex.reader().search("shame on you")
    generated_hits = [r for r in after if r.generated]
    assert len(generated_hits) == 1
    await lex.engine().generate(exact[1])  # the second suggestion → cache hit
    await lex.engine().generate(generated_hits[0])
    assert gen.calls == 1  # nothing regenerated

    async with session_scope(session_factory) as session:
        rows = (
            (await session.execute(select(Word).where(Word.match_key == "shame on you")))
            .scalars()
            .all()
        )
    assert len(rows) == 1


async def test_surface_variants_converge(engine):
    # One Cambridge word reachable by three surface forms → one entry.
    lex, gen, session_factory = _make_lexicon(
        engine,
        cam_words={
            5: (
                "look after someone/something",
                "phrasal_verb",
                ["look after", "look after somebody", "look after sb"],
            )
        },
        norm_by_id={5: "look after {sb}"},
        results_by_word={
            match_key("look after {sb}"): GeneratedResult(units=[_entry("look after {sb}")])
        },
    )
    norms = set()
    for variant in ("look after", "look after somebody", "look after sb"):
        hit = (await lex.reader().search(variant))[0]
        norms.add((await lex.engine().generate(hit)).norm)
    assert norms == {"look after {sb}"}
    assert gen.calls == 1
    async with session_scope(session_factory) as session:
        rows = (
            (await session.execute(select(Word).where(Word.cambridge_word_id == 5))).scalars().all()
        )
    assert len(rows) == 1


async def test_concurrent_generate_once(engine):
    lex, gen, _sf = _make_lexicon(
        engine,
        cam_words={7: ("book", "word", ["book"])},
        norm_by_id={7: "book"},
        results_by_word={match_key("book"): GeneratedResult(units=[_entry("book")])},
    )
    hit = (await lex.reader().search("book"))[0]
    entries = await asyncio.gather(*[lex.engine().generate(hit) for _ in range(5)])
    assert all(isinstance(e, Entry) for e in entries)
    assert gen.calls == 1  # per-key lock + double-check


async def test_lock_dict_does_not_grow(engine):
    lex, gen, _sf = _make_lexicon(
        engine,
        cam_words={i: (w, "word", [w]) for i, w in enumerate(("alpha", "beta", "gamma"), 1)},
        norm_by_id={i: w for i, w in enumerate(("alpha", "beta", "gamma"), 1)},
        results_by_word={
            match_key(w): GeneratedResult(units=[_entry(w)]) for w in ("alpha", "beta", "gamma")
        },
    )
    hits = {w: (await lex.reader().search(w))[0] for w in ("alpha", "beta", "gamma")}
    for w in ("alpha", "beta", "gamma"):
        await lex.engine().generate(hits[w])
    await asyncio.gather(*[lex.engine().generate(hits["alpha"]) for _ in range(4)])
    assert gen.calls == 3
    assert len(lex._locks) == 0  # every lock evicted


async def test_get_and_status_by_lexi_id(engine):
    lex, gen, _sf = _make_lexicon(
        engine,
        cam_words={1: ("color", "word", ["color"])},
        norm_by_id={1: "color"},
        results_by_word={match_key("color"): GeneratedResult(units=[_entry("color")])},
    )
    hit = (await lex.reader().search("color"))[0]
    entry = await lex.engine().generate(hit)
    lexi_id = (await lex.reader().search("color"))[0].lexi_word_id
    assert lexi_id is not None

    fetched = await lex.reader().get_entry(lexi_id)
    assert fetched.norm == entry.norm
    assert await lex.reader().get_status(lexi_id) == "done"
    # Unknown id → None, no crash.
    assert await lex.reader().get_status(999999) is None


async def test_search_finds_custom_word(engine):
    # A custom word (not in Cambridge) is still findable by search after add.
    lex, gen, _sf = _make_lexicon(
        engine,
        cam_words={},
        norm_by_id={},
        results_by_word={
            match_key("doomscrolling"): GeneratedResult(units=[_entry("doomscrolling")])
        },
    )
    entry = await lex.engine().generate("doomscrolling")
    assert entry.norm == "doomscrolling"
    assert gen.calls == 1
    async with session_scope(lex._session_factory) as session:
        w = (
            await session.execute(select(Word).where(Word.match_key == match_key("doomscrolling")))
        ).scalar_one()
        assert w.cambridge_word_id is None


async def test_generate_custom_no_overwrite(engine):
    lex, gen, session_factory = _make_lexicon(
        engine,
        cam_words={},
        norm_by_id={},
        results_by_word={
            match_key("doomscrolling"): GeneratedResult(units=[_entry("doomscrolling")])
        },
    )
    first = await lex.engine().generate("doomscrolling")
    assert gen.calls == 1
    async with session_scope(session_factory) as session:
        before = (
            await session.execute(select(Word).where(Word.match_key == match_key("doomscrolling")))
        ).scalar_one()
        updated_before = before.updated_at

    # Same custom string again: cache-first, no regeneration, row untouched.
    second = await lex.engine().generate("doomscrolling")
    assert second.norm == first.norm
    assert gen.calls == 1
    async with session_scope(session_factory) as session:
        after = (
            await session.execute(select(Word).where(Word.match_key == match_key("doomscrolling")))
        ).scalar_one()
        assert after.updated_at == updated_before


async def test_force_regenerates(engine):
    lex, gen, _sf = _make_lexicon(
        engine,
        cam_words={1: ("color", "word", ["color"])},
        norm_by_id={1: "color"},
        results_by_word={match_key("color"): GeneratedResult(units=[_entry("color")])},
    )
    hit = (await lex.reader().search("color"))[0]
    await lex.engine().generate(hit)
    assert gen.calls == 1
    # Without force: cache hit. With force: regenerate.
    await lex.engine().generate((await lex.reader().search("color"))[0])
    assert gen.calls == 1
    await lex.engine().generate((await lex.reader().search("color"))[0], force=True)
    assert gen.calls == 2


async def test_display_is_rendered_from_norm(engine):
    lex, gen, _sf = _make_lexicon(
        engine,
        cam_words={9: ("act on behalf of somebody", "expression", ["act on behalf of somebody"])},
        norm_by_id={9: "act on behalf of {sb}"},
        results_by_word={
            match_key("act on behalf of {sb}"): GeneratedResult(
                units=[_entry("act on behalf of {sb}")]
            )
        },
    )
    hit = (await lex.reader().search("act on behalf of somebody"))[0]
    entry = await lex.engine().generate(hit)
    assert entry.display == "act on behalf of somebody"
    assert entry.norm == "act on behalf of {sb}"


async def test_senses_sorted_by_tier(engine):
    multi = GeneratedEntry(
        norm="run",
        entry_type="word",
        senses=[
            {"definition": "rare meaning", "tier": "rare", "pos": "verb"},
            {"definition": "core meaning", "tier": "core", "pos": "verb"},
            {"definition": "common meaning", "tier": "common", "pos": "verb"},
        ],
    )
    lex, gen, _sf = _make_lexicon(
        engine,
        cam_words={3: ("run", "word", ["run"])},
        norm_by_id={3: "run"},
        results_by_word={match_key("run"): GeneratedResult(units=[multi])},
    )
    hit = (await lex.reader().search("run"))[0]
    entry = await lex.engine().generate(hit)
    assert [s.tier for s in entry.senses] == ["core", "common", "rare"]


async def test_split_page_units_persist(engine):
    split = GeneratedResult(units=[_entry("idiom one"), _entry("idiom two")])
    lex, gen, session_factory = _make_lexicon(
        engine,
        cam_words={20: ("idiom one", "idiom", ["idiom one"])},
        norm_by_id={20: "idiom one"},
        results_by_word={match_key("idiom one"): split},
    )
    hit = (await lex.reader().search("idiom one"))[0]
    entry = await lex.engine().generate(hit)
    assert isinstance(entry, Entry)
    assert gen.calls == 1
    async with session_scope(session_factory) as session:
        rows = (
            (await session.execute(select(Word).where(Word.cambridge_word_id == 20)))
            .scalars()
            .all()
        )
    assert {r.norm for r in rows} == {"idiom one", "idiom two"}


async def test_idiom_is_first_class_learnable_content(engine):
    # Characterization: what happens today when an idiom is generated and how it
    # is reached from a host word. Documents the pre-change contract so a later
    # change to the discovery surface is a deliberate, visible edit.
    #
    # A host word ("kick") declares a related idiom ("kick the bucket"); the
    # generator returns it as an idiom-typed related link. Generation persists a
    # `part_of_phrasal_family`-style pointer to a pending stub for the idiom.
    host = GeneratedEntry(
        norm="kick",
        entry_type="word",
        pos="verb",
        senses=[
            {
                "definition": "strike with the foot",
                "tier": "core",
                "cefr_level": "A1",
                "pos": "verb",
            }
        ],
        related=[RelatedWord(norm="kick the bucket", rel_type="part_of_phrasal_family")],
    )
    lex, gen, session_factory = _make_lexicon(
        engine,
        cam_words={30: ("kick", "word", ["kick"])},
        norm_by_id={30: "kick"},
        results_by_word={
            match_key("kick"): GeneratedResult(units=[host]),
            # The idiom, generated on its own, carries its idiom entry_type and
            # a full core sense — it is NOT a second-class link target.
            match_key("kick the bucket"): GeneratedResult(
                units=[
                    GeneratedEntry(
                        norm="kick the bucket",
                        entry_type="idiom",
                        pos=None,
                        senses=[
                            {
                                "definition": "to die",
                                "tier": "core",
                                "cefr_level": "B2",
                                "pos": "verb",
                            }
                        ],
                    )
                ]
            ),
        },
    )

    host_entry = await lex.engine().generate((await lex.reader().search("kick"))[0])

    # The idiom surfaces on the host as a relation link carrying a generatable
    # handle: display, rel_type, plus the linked word's id + status.
    idiom_links = [link for link in host_entry.links if link.norm == "kick the bucket"]
    assert len(idiom_links) == 1
    assert idiom_links[0].rel_type == "part_of_phrasal_family"
    assert idiom_links[0].status == "pending"

    # The idiom already exists as a pending, browsable stub (lazy-generation
    # queue): it shows up in the whole-dictionary browse filtered to pending.
    pending = await lex.reader().list_entries(status="pending")
    stub_hits = [r for r in pending if r.display == "kick the bucket"]
    assert len(stub_hits) == 1
    assert stub_hits[0].lexi_word_id is not None

    # TODAY: a related-seeded stub carries NO Cambridge provenance, so search()
    # (which resolves through Cambridge) does not surface it. The stub is reached
    # by generating its norm as a custom string — which converges on the existing
    # stub row by match_key and yields FULL sense content, keeping entry_type.
    # It is a first-class entry, not merely a pointer.
    assert await lex.reader().search("kick the bucket") == []
    idiom_entry = await lex.engine().generate("kick the bucket")
    assert idiom_entry.entry_type == "idiom"
    assert idiom_entry.norm == "kick the bucket"
    assert [s.definition for s in idiom_entry.senses] == ["to die"]
    assert idiom_entry.senses[0].tier == "core"


async def test_entry_link_carries_generatable_handle(engine):
    # A host link must expose the target's word_id + status, so a consumer can go
    # straight to get_entry (done) or generate (pending) without a search round-trip
    # (a related-seeded stub has no Cambridge provenance, so search never finds it).
    host = GeneratedEntry(
        norm="kick",
        entry_type="word",
        pos="verb",
        senses=[
            {
                "definition": "strike with the foot",
                "tier": "core",
                "cefr_level": "A1",
                "pos": "verb",
            }
        ],
        related=[RelatedWord(norm="kick the bucket", rel_type="part_of_phrasal_family")],
    )
    lex, _gen, _sf = _make_lexicon(
        engine,
        cam_words={30: ("kick", "word", ["kick"])},
        norm_by_id={30: "kick"},
        results_by_word={
            match_key("kick"): GeneratedResult(units=[host]),
            match_key("kick the bucket"): GeneratedResult(
                units=[
                    GeneratedEntry(
                        norm="kick the bucket",
                        entry_type="idiom",
                        pos=None,
                        senses=[
                            {
                                "definition": "to die",
                                "tier": "core",
                                "cefr_level": "B2",
                                "pos": "verb",
                            }
                        ],
                    )
                ]
            ),
        },
    )

    host_entry = await lex.engine().generate((await lex.reader().search("kick"))[0])
    (link,) = [ln for ln in host_entry.links if ln.norm == "kick the bucket"]
    # The pending idiom is reachable by its handle, not by search.
    assert link.status == "pending"
    idiom_entry = await lex.engine().generate("kick the bucket")

    # After generation the same link reports done, and its word_id addresses the
    # generated entry directly (get_entry, no LLM, no search).
    host_again = await lex.reader().get_entry(host_entry.word_id)
    (link2,) = [ln for ln in host_again.links if ln.norm == "kick the bucket"]
    assert link2.status == "done"
    assert link2.word_id == idiom_entry.word_id
    fetched = await lex.reader().get_entry(link2.word_id)
    assert fetched.entry_type == "idiom"
    assert [s.definition for s in fetched.senses] == ["to die"]


# --- embeddings + semantic search ----------------------------------------

_VOCAB = "abcdefghijklmnopqrstuvwxyz "


def _bag_encode(texts: list[str]) -> list[list[float]]:
    """Deterministic char-bag embedder, L2-normalized — no torch, assertable.

    Texts sharing more characters (a definition and a semantically-overlapping
    query) get a higher cosine, so ranking is meaningful without a real model.
    """
    import math

    out: list[list[float]] = []
    for t in texts:
        v = [0.0] * len(_VOCAB)
        for ch in t.lower():
            i = _VOCAB.find(ch)
            if i >= 0:
                v[i] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / norm for x in v])
    return out


def _fake_embedder(model_name: str = "fake-v1") -> Embedder:
    return Embedder(encode=_bag_encode, model_name=model_name, dim=len(_VOCAB))


def _def_entry(norm: str, definition: str) -> GeneratedEntry:
    return GeneratedEntry(
        norm=norm,
        entry_type="word",
        pos="noun",
        senses=[{"definition": definition, "tier": "core", "pos": "noun"}],
    )


def _pet_lexicon(engine, embedder, vectors=None):
    return _make_lexicon(
        engine,
        cam_words={
            1: ("dog", "word", ["dog"]),
            2: ("cat", "word", ["cat"]),
            3: ("automobile", "word", ["automobile"]),
        },
        norm_by_id={1: "dog", 2: "cat", 3: "automobile"},
        results_by_word={
            match_key("dog"): GeneratedResult(
                units=[_def_entry("dog", "a loyal four-legged pet animal that barks")]
            ),
            match_key("cat"): GeneratedResult(
                units=[_def_entry("cat", "a small furry pet animal that meows")]
            ),
            match_key("automobile"): GeneratedResult(
                units=[_def_entry("automobile", "a road vehicle powered by an engine")]
            ),
        },
        embedder=embedder,
        vectors=vectors,
    )


async def test_generate_embeds_each_sense(engine):
    lex, _gen, _sf = _pet_lexicon(engine, _fake_embedder())
    for w in ("dog", "cat", "automobile"):
        await lex.engine().generate((await lex.reader().search(w))[0])

    # Vectors live in the index, tagged with the encoder that made them.
    assert len(await lex._vectors.ids()) == 3
    assert await lex._vectors.ids({"model": "fake-v1"}) == await lex._vectors.ids()
    assert all(len(v) == len(_VOCAB) for v in (await lex._vectors.fetch(["1", "2", "3"])).values())


async def test_generate_survives_embedder_error(engine):
    # An embedder that raises a NON-EmbeddingUnavailable error at encode time
    # (e.g. CUDA OOM, bad model, device error) must NOT fail the already-paid
    # generation: the word persists done with no vector, best-effort.
    def boom(_texts):
        raise RuntimeError("CUDA out of memory")

    embedder = Embedder(encode=boom, model_name="boom", dim=8)
    lex, gen, _sf = _pet_lexicon(engine, embedder)
    entry = await lex.engine().generate((await lex.reader().search("dog"))[0])
    assert entry.display == "dog"
    assert gen.calls == 1
    assert await lex._vectors.ids() == set()  # embed failed, generation succeeded
    # semantic_search also degrades to [] rather than raising on the same embedder.
    assert await lex.reader().semantic_search("pet") == []


async def test_generate_survives_an_unreachable_vector_index(engine):
    """A dead index must not fail a paid generation, and must not fail a read."""

    class DeadIndex:
        async def upsert(self, records):
            raise RuntimeError("index unreachable")

        async def query(self, vector, k, where=None):
            raise RuntimeError("index unreachable")

        async def delete(self, ids):
            raise RuntimeError("index unreachable")

        async def ids(self, where=None):
            raise RuntimeError("index unreachable")

        async def fetch(self, ids):
            raise RuntimeError("index unreachable")

    lex, _gen, _sf = _pet_lexicon(engine, _fake_embedder(), vectors=DeadIndex())
    entry = await lex.engine().generate((await lex.reader().search("dog"))[0])

    assert entry.display == "dog"
    assert await lex.reader().semantic_search("pet") == []
    assert await lex.engine().backfill_embeddings() == 0


async def test_semantic_search_ranks_by_meaning(engine):
    lex, _gen, _sf = _pet_lexicon(engine, _fake_embedder())
    for w in ("dog", "cat", "automobile"):
        await lex.engine().generate((await lex.reader().search(w))[0])
    hits = await lex.reader().semantic_search("a pet animal", k=3)
    assert len(hits) == 3
    # Pet definitions must rank above the vehicle one; scores are non-increasing.
    assert hits[-1].display == "automobile"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert all(isinstance(h.lexi_word_id, int) for h in hits)
    # The hit's word id resolves via the normal read path.
    top = await lex.reader().get_entry(hits[0].lexi_word_id)
    assert top.display == hits[0].display


async def test_semantic_search_respects_k(engine):
    lex, _gen, _sf = _pet_lexicon(engine, _fake_embedder())
    for w in ("dog", "cat", "automobile"):
        await lex.engine().generate((await lex.reader().search(w))[0])
    assert len(await lex.reader().semantic_search("pet", k=1)) == 1
    assert await lex.reader().semantic_search("pet", k=0) == []


async def test_semantic_search_empty_when_nothing_embedded(engine):
    # No embedder available (extra missing): generation still works (best-effort),
    # nothing is indexed, and semantic_search returns [] rather than raising.
    lex, _gen, _sf = _pet_lexicon(engine, embedder=None)
    await lex.engine().generate((await lex.reader().search("dog"))[0])
    assert await lex._vectors.ids() == set()
    assert await lex.reader().semantic_search("pet") == []


async def test_semantic_search_drops_a_vector_whose_sense_is_gone(engine):
    """A stale vector cannot resurrect a deleted sense — it is skipped on hydration."""
    lex, _gen, _sf = _pet_lexicon(engine, _fake_embedder())
    for w in ("dog", "cat"):
        await lex.engine().generate((await lex.reader().search(w))[0])
    await lex._vectors.upsert(
        [VectorRecord(id="4242", vector=[1.0] * len(_VOCAB), meta={"model": "fake-v1"})]
    )

    hits = await lex.reader().semantic_search("pet", k=3)

    assert {hit.display for hit in hits} == {"dog", "cat"}


async def test_deleting_an_entry_forgets_its_vectors(engine):
    lex, _gen, _sf = _pet_lexicon(engine, _fake_embedder())
    entry = await lex.engine().generate((await lex.reader().search("dog"))[0])
    assert await lex._vectors.ids() != set()

    assert await lex.engine().delete_entry(entry.word_id) is True

    assert await lex._vectors.ids() == set()


async def test_backfill_fills_then_idempotent(engine):
    # Generate with no embedder (nothing indexed), then 'install' one and backfill.
    lex, _gen, _sf = _pet_lexicon(engine, embedder=None)
    for w in ("dog", "cat", "automobile"):
        await lex.engine().generate((await lex.reader().search(w))[0])
    lex._embedder = _fake_embedder()
    assert await lex.engine().backfill_embeddings() == 3
    assert len(await lex._vectors.ids()) == 3
    # Everything embedded now → second backfill is a no-op.
    assert await lex.engine().backfill_embeddings() == 0
    # And semantic search now works.
    assert len(await lex.reader().semantic_search("pet", k=2)) == 2


async def test_backfill_prunes_vectors_whose_sense_is_gone(engine):
    """The reconciliation step the eventually-consistent index depends on."""
    lex, _gen, _sf = _pet_lexicon(engine, _fake_embedder())
    await lex.engine().generate((await lex.reader().search("dog"))[0])
    await lex._vectors.upsert(
        [VectorRecord(id="4242", vector=[1.0] * len(_VOCAB), meta={"model": "fake-v1"})]
    )

    await lex.engine().backfill_embeddings()

    assert "4242" not in await lex._vectors.ids()


async def test_backfill_reembeds_on_model_change(engine):
    lex, _gen, _sf = _pet_lexicon(engine, _fake_embedder("m1"))
    await lex.engine().generate((await lex.reader().search("dog"))[0])
    # Switch model: the m1 vector is stale, so search under m2 sees nothing…
    lex._embedder = _fake_embedder("m2")
    assert await lex.reader().semantic_search("pet") == []
    # …until backfill re-embeds it under the new model.
    assert await lex.engine().backfill_embeddings() == 1
    hits = await lex.reader().semantic_search("pet")
    assert len(hits) == 1 and hits[0].display == "dog"
    # Re-embedding replaces the vector in place rather than accumulating one per model.
    assert len(await lex._vectors.ids()) == 1


async def test_backfill_limit(engine):
    lex, _gen, _sf = _pet_lexicon(engine, embedder=None)
    for w in ("dog", "cat", "automobile"):
        await lex.engine().generate((await lex.reader().search(w))[0])
    lex._embedder = _fake_embedder()
    assert await lex.engine().backfill_embeddings(limit=2) == 2
    assert await lex.engine().backfill_embeddings() == 1  # the remaining one


def _entry_with_topics(norm, topics) -> GeneratedEntry:
    return GeneratedEntry(
        norm=norm,
        entry_type="word",
        pos="noun",
        senses=[{"definition": f"def of {norm}", "tier": "core", "pos": "noun"}],
        topics=[{"tag": t, "title": ti} for t, ti in topics],
    )


async def test_generate_exposes_topics_and_injects_vocab(engine):
    lex, gen, _sf = _make_lexicon(
        engine,
        cam_words={1: ("bank", "word", ["bank"]), 2: ("apple", "word", ["apple"])},
        norm_by_id={1: "bank", 2: "apple"},
        results_by_word={
            match_key("bank"): GeneratedResult(
                units=[_entry_with_topics("bank", [("business", "Business & Finance")])]
            ),
            match_key("apple"): GeneratedResult(
                units=[_entry_with_topics("apple", [("food", "Food & Drink")])]
            ),
        },
    )
    entry = await lex.engine().generate((await lex.reader().search("bank"))[0])
    assert [(t.name, t.title) for t in entry.topics] == [("business", "Business & Finance")]

    # The second generation must SEE the existing vocab injected for reuse.
    await lex.engine().generate((await lex.reader().search("apple"))[0])
    assert ("business", "Business & Finance") in gen.last_existing_tags


async def test_get_senses_resolves_by_id(engine):
    lex, _gen, _sf = _make_lexicon(
        engine,
        cam_words={1: ("color", "word", ["color"])},
        norm_by_id={1: "color"},
        results_by_word={match_key("color"): GeneratedResult(units=[_entry("color")])},
    )
    entry = await lex.engine().generate((await lex.reader().search("color"))[0])
    sense_ids = [s.sense_id for s in entry.senses]
    assert sense_ids and all(sid is not None for sid in sense_ids)

    views = await lex.reader().get_senses(sense_ids)
    assert len(views) == len(sense_ids)
    assert views[0].definition == "def of color"
    assert views[0].sense_id == sense_ids[0]


async def test_get_senses_empty_and_missing(engine):
    lex, _gen, _sf = _make_lexicon(
        engine,
        cam_words={1: ("color", "word", ["color"])},
        norm_by_id={1: "color"},
        results_by_word={match_key("color"): GeneratedResult(units=[_entry("color")])},
    )
    assert await lex.reader().get_senses([]) == []
    # Unknown ids are skipped, not errored.
    assert await lex.reader().get_senses([999999]) == []


# --- add_examples (targeted neutral augmentation) -------------------------


async def _seed_sense_with_examples(engine, existing: list[str], embedder=None):
    """Seed a done word with one sense carrying ``existing`` example texts.

    Returns ``(lex, gen, session_factory, sense_id)``. The lexicon's fake
    generator carries a canned two-example ExampleBatch for the augment path."""
    lex, gen, session_factory = _make_lexicon(
        engine,
        cam_words={1: ("color", "word", ["color"])},
        norm_by_id={1: "color"},
        results_by_word={match_key("color"): GeneratedResult(units=[_entry("color")])},
        embedder=embedder,
        example_batch=ExampleBatch(
            examples=[
                'She <t inf="past">painted</t> the fence a bright color.',
                'The <t inf="plural">colors</t> of autumn are stunning.',
            ]
        ),
    )
    entry = await lex.engine().generate((await lex.reader().search("color"))[0])
    sense_id = entry.senses[0].sense_id
    async with session_scope(session_factory) as session:
        for order, text in enumerate(existing):
            session.add(Example(sense_id=sense_id, text=text, example_order=order))
        await session.flush()
    return lex, gen, session_factory, sense_id


async def test_add_examples_appends_without_touching_existing(engine):
    lex, gen, session_factory, sense_id = await _seed_sense_with_examples(
        engine, ["An existing example.", "A second one."]
    )
    view = await lex.engine().add_examples(sense_id, n=2)
    # Old two kept, two new appended, order contiguous.
    assert view.examples[:2] == ["An existing example.", "A second one."]
    assert len(view.examples) == 4
    async with session_scope(session_factory) as session:
        rows = (
            (
                await session.execute(
                    select(Example.example_order).where(Example.sense_id == sense_id)
                )
            )
            .scalars()
            .all()
        )
    assert sorted(rows) == [0, 1, 2, 3]


async def test_add_examples_returns_sense_view_with_all_examples(engine):
    lex, _gen, _sf, sense_id = await _seed_sense_with_examples(engine, ["Old."])
    view = await lex.engine().add_examples(sense_id, n=2)
    assert isinstance(view, SenseView)
    assert view.sense_id == sense_id
    assert len(view.examples) == 3


async def test_add_examples_feeds_existing_to_generator(engine):
    lex, gen, _sf, sense_id = await _seed_sense_with_examples(
        engine, ["First existing.", "Second existing."]
    )
    await lex.engine().add_examples(sense_id, n=2)
    # Soft dedup: the generator saw the existing examples + the requested n.
    assert gen.last_existing == ["First existing.", "Second existing."]
    assert gen.last_n == 2


async def test_add_examples_new_examples_carry_parseable_tags(engine):
    lex, _gen, _sf, sense_id = await _seed_sense_with_examples(engine, [])
    view = await lex.engine().add_examples(sense_id, n=2)
    # The appended examples carry <t inf> tags the markup reader can parse.
    tagged = [e for e in view.examples if "<t inf=" in e]
    assert len(tagged) == 2
    for text in tagged:
        clean, spans = parse_marked_example(text)
        assert spans and spans[0].surface  # a target span was extracted
        assert "<t" not in clean  # the clean form has tags unwrapped


async def test_add_examples_zero_is_noop(engine):
    lex, gen, _sf, sense_id = await _seed_sense_with_examples(engine, ["Only one."])
    view = await lex.engine().add_examples(sense_id, n=0)
    assert view.examples == ["Only one."]
    assert gen.example_calls == 0  # no LLM call for n<=0


async def test_add_examples_clamps_n_to_schema_ceiling(engine):
    # n above ExampleBatch's 12-item ceiling is clamped so the model is never
    # prompted for more than it can validly return (avoids a schema-reject retry).
    lex, gen, _sf, sense_id = await _seed_sense_with_examples(engine, [])
    await lex.engine().add_examples(sense_id, n=100)
    assert gen.last_n == 12


async def test_add_examples_unknown_sense_raises(engine):
    lex, _gen, _sf = _make_lexicon(
        engine,
        cam_words={1: ("color", "word", ["color"])},
        norm_by_id={1: "color"},
        results_by_word={match_key("color"): GeneratedResult(units=[_entry("color")])},
    )
    with pytest.raises(ValueError):
        await lex.engine().add_examples(999999, n=2)


async def test_add_examples_does_not_reembed(engine):
    # A word generated with a real fake embedder is embedded once at generation;
    # add_examples must not trigger a re-embed (embeddings cover the definition only).
    embedder = _fake_embedder()
    lex, _gen, _sf, sense_id = await _seed_sense_with_examples(
        engine, ["Old."], embedder=embedder
    )
    before = await lex._vectors.fetch([str(sense_id)])

    await lex.engine().add_examples(sense_id, n=2)

    assert await lex._vectors.fetch([str(sense_id)]) == before


# --- stats (read-only counts) ---------------------------------------------


async def test_stats_matches_seeded_fixture(engine):
    from lexi_ai.infrastructure.db.models import (
        Asset,
        Example,
        Question,
        Tag,
        Theme,
        ThemedExample,
        ThemedSense,
        Word,
        WordTag,
    )

    lex, _gen, session_factory = _make_lexicon(
        engine, cam_words={}, norm_by_id={}, results_by_word={}
    )
    async with session_scope(session_factory) as session:
        # Two done words (one with 2 senses/3 examples, one with 1 sense/1 example),
        # a pending stub, and an error word.
        w1 = Word(norm="alpha", match_key="alpha", status="done")
        w2 = Word(norm="beta", match_key="beta", status="done")
        w3 = Word(norm="gamma", match_key="gamma", status="pending")
        w4 = Word(norm="delta", match_key="delta", status="error")
        session.add_all([w1, w2, w3, w4])
        await session.flush()
        s1 = Sense(word_id=w1.id, definition="d1", tier="core", sense_order=0)
        s2 = Sense(word_id=w1.id, definition="d2", tier="common", sense_order=1)
        s3 = Sense(word_id=w2.id, definition="d3", tier="core", sense_order=0)
        session.add_all([s1, s2, s3])
        await session.flush()
        session.add_all(
            [
                Example(sense_id=s1.id, text="e1", example_order=0),
                Example(sense_id=s1.id, text="e2", example_order=1),
                Example(sense_id=s3.id, text="e3", example_order=0),
            ]
        )
        # One tag linked to w1; one theme with an overlay on s1 (one themed word).
        tag = Tag(name="business", title="Business", tag_key="business")
        theme = Theme(theme_key="bard", name="Bard", style_prompt="voice")
        session.add_all([tag, theme])
        await session.flush()
        session.add(WordTag(word_id=w1.id, tag_id=tag.id))
        ts = ThemedSense(sense_id=s1.id, theme_id=theme.id, definition="themed d1")
        session.add(ts)
        await session.flush()
        session.add(ThemedExample(themed_sense_id=ts.id, text="te1", example_order=0))
        # One asset of each kind + one question.
        session.add_all(
            [
                Asset(
                    source_kind="sense_def",
                    source_id=s1.id,
                    kind="translate",
                    params="vi",
                    content_hash="h1",
                    text_value="dịch",
                ),
                Asset(
                    source_kind="sense_def",
                    source_id=s1.id,
                    kind="tts",
                    params="alloy|mp3",
                    content_hash="h2",
                    file_path="x.mp3",
                ),
                Question(
                    word_id=w1.id,
                    sense_id=s1.id,
                    type_id="contextual_mcq",
                    render_format="single_choice",
                    difficulty_level=1,
                    interaction_mode="assessment",
                    payload="{}",
                    content_hash="question-hash",
                ),
            ]
        )
        await session.flush()

    stats = await lex.reader().stats()
    assert stats.words_by_status == {"done": 2, "pending": 1, "error": 1}
    assert stats.senses == 3
    assert stats.examples == 3
    assert stats.tags == 1
    assert stats.themes == 1
    assert stats.themed_words == 1  # only w1 has a themed overlay
    assert stats.assets_by_kind == {"translate": 1, "tts": 1}
    assert stats.questions == 1


async def test_stats_empty_dictionary(engine):
    lex, _gen, _sf = _make_lexicon(engine, cam_words={}, norm_by_id={}, results_by_word={})
    stats = await lex.reader().stats()
    assert stats.words_by_status == {}
    assert stats.senses == 0
    assert stats.examples == 0
    assert stats.tags == 0
    assert stats.themes == 0
    assert stats.themed_words == 0
    assert stats.assets_by_kind == {}
    assert stats.questions == 0


# --- public question API -------------------------------------------------


def test_question_public_exports_and_score_is_internal():
    import lexi_ai
    from lexi_ai import (
        AnswerSubmission,
        Evaluation,
        PrepareDemand,
        PresentedQuestion,
        QuestionTypeInfo,
    )

    assert Evaluation.__name__ == "Evaluation"
    assert PresentedQuestion.__name__ == "PresentedQuestion"
    assert AnswerSubmission.__name__ == "AnswerSubmission"
    assert PrepareDemand.__name__ == "PrepareDemand"
    assert QuestionTypeInfo.__name__ == "QuestionTypeInfo"
    # The legacy read-model / descriptor names are no longer public.
    assert not hasattr(lexi_ai, "Score")
    assert not hasattr(lexi_ai, "Question")
    assert not hasattr(lexi_ai, "QuestionDemand")
    assert not hasattr(lexi_ai, "QuestionTypeDescriptor")


def test_the_question_surface_lives_on_the_facades_not_the_composition_root():
    expected = {
        "question_types",
        "get_question",
        "list_questions_for_sense",
        "retrieve_question",
        "retrieve_exposure",
        "evaluate_answer",
    }
    assert all(hasattr(LexiconReader, name) for name in expected)
    assert all(hasattr(LexiconEngine, name) for name in expected | {"prepare_questions"})
    # The composition root wires the contexts; it does not serve questions itself.
    assert not any(hasattr(Lexicon, name) for name in expected | {"prepare_questions"})
    assert not any(
        hasattr(LexiconEngine, name) for name in ("generate_questions_for_sense", "grade_question")
    )


async def test_reader_and_worker_question_engines_have_separate_judge_contexts(engine):
    lex, _gen, _sf = _make_lexicon(engine, cam_words={}, norm_by_id={}, results_by_word={})
    judge = object()
    lex._providers.judge_llm = lambda: judge

    reader_engine = lex._question_engines.engine(providers=False)
    worker_engine = lex._question_engines.engine(providers=True)

    assert reader_engine is not worker_engine
    assert reader_engine._judge is None
    assert worker_engine._judge is judge


class _QuestionRepositorySpy:
    def __init__(self, question):
        self.question = question
        self.requested_ids = []

    async def get(self, question_id):
        self.requested_ids.append(question_id)
        return self.question

    async def list_for_sense(self, sense_id, type_id=None):
        return [self.question]


class _QuestionEngineSpy:
    def __init__(self):
        self.evaluated = []

    async def evaluate(self, question, submission):
        from lexi_ai.contracts.questions import Evaluation

        self.evaluated.append((question, submission))
        return Evaluation(
            question_id=submission.question_id, status="graded", correct=True, score=1.0
        )


async def test_evaluate_answer_refetches_authoritative_question_by_public_id(engine):
    from lexi_ai.contracts.questions import AnswerSubmission, ChoiceResponse, RenderKind
    from lexi_ai.domain.questions import PersistedQuestion

    authoritative = PersistedQuestion(
        question_id=41,
        word_id=3,
        sense_id=7,
        type_id="definition_mcq",
        render_kind=RenderKind.SINGLE_CHOICE,
        difficulty_level=1,
        interaction="assessment",
        payload={"stem": "?", "options": ["eloquent"], "correct_index": 0},
    )
    lex, _gen, _sf = _make_lexicon(engine, cam_words={}, norm_by_id={}, results_by_word={})
    repository = _QuestionRepositorySpy(authoritative)
    question_engine = _QuestionEngineSpy()
    lex._question_engines._repo = repository
    lex._question_engines.worker = question_engine

    submission = AnswerSubmission(question_id="41", response=ChoiceResponse(selected_index=0))
    evaluation = await lex.engine().evaluate_answer(41, submission)

    assert evaluation.status == "graded"
    assert repository.requested_ids == [41]
    assert question_engine.evaluated == [(authoritative, submission)]


async def test_evaluate_answer_returns_none_for_unknown_question(engine):
    from lexi_ai.contracts.questions import AnswerSubmission, TextResponse

    lex, _gen, _sf = _make_lexicon(engine, cam_words={}, norm_by_id={}, results_by_word={})
    lex._question_engines._repo = _QuestionRepositorySpy(None)
    question_engine = _QuestionEngineSpy()
    lex._question_engines.worker = question_engine

    submission = AnswerSubmission(question_id="999", response=TextResponse(text="answer"))
    assert await lex.engine().evaluate_answer(999, submission) is None
    assert question_engine.evaluated == []


async def test_close_disposes_owned_engine():
    class DisposableEngine:
        def __init__(self) -> None:
            self.disposed = False

        async def dispose(self) -> None:
            self.disposed = True

    engine = DisposableEngine()
    lexicon = Lexicon(None, None, None, engine=engine)  # type: ignore[arg-type]

    await lexicon.close()

    assert engine.disposed
