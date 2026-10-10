"""Unified search owns lexical matching; generation consumes classified hits."""

import asyncio
import sqlite3

import pytest
import tantivy
from pydantic import TypeAdapter, ValidationError
from test_datasets import dictionary as dictionary
from test_prompting import prompt_context
from test_questions_from_word import LLM as QuestionLLM
from test_word_generation import payload, stage_payload

from lexi_ai import Lexicon, MatchKind, QuestionType
from lexi_ai.errors import InvalidResourceError
from lexi_ai.models import SearchResult
from lexi_ai.references.cambridge import SourceHit, encode_reference_id
from lexi_ai.schema import Base, Sense, SenseForm, Word, WordAlias
from lexi_ai.words.search import Search


@pytest.fixture
async def sqlite_catalog(tmp_path):
    lexicon = Lexicon(f"sqlite+aiosqlite:///{tmp_path / 'content.sqlite'}")
    try:
        await lexicon.db.create_schema(Base.metadata)
        yield lexicon
    finally:
        await lexicon.close()


@pytest.fixture(params=["sqlite", "postgresql"])
def catalog(request):
    return request.getfixturevalue("sqlite_catalog" if request.param == "sqlite" else "dictionary")


class Transport:
    def __init__(self):
        self.calls = []
        self.questions = QuestionLLM()

    async def complete(self, instruction, data, schema):
        self.calls.append(schema.__name__)
        if schema.__name__ == "ThemeParts":
            return schema(voice="Coach", diction="sport")
        if schema.__name__ == "ThemedWord":
            count = prompt_context(data, "generation_parameters")["examples_per_sense"]
            return schema(
                senses=[
                    {
                        "definition": "Move quickly during training",
                        "examples": [f"I [run] for {i} minutes." for i in range(count)],
                    }
                ]
            )
        if schema.__name__ in {"QuestionBatch", "AnchoredQuestionBatch"}:
            return await self.questions.complete(instruction, data, schema)
        output = payload("run")
        output["aliases"] = ["sprint"]
        output["senses"][0].update(
            definition="Move quickly on foot",
            pos="VERB",
            examples=["I [run] every day."],
            forms=["running|ing"],
            patterns=["run {sth}"],
        )
        return stage_payload(output, data, schema)


async def prepare_run(catalog, source):
    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE words SET word='run',display_form='run'")
        connection.execute("UPDATE entries SET pos='verb'")
        connection.execute("UPDATE senses SET definition='Move quickly on foot'")
        connection.execute("INSERT INTO words VALUES(2,'running','running','word','done')")
        connection.execute("INSERT INTO entries VALUES(12,2,'noun',0,NULL,NULL,NULL)")
        connection.execute(
            "INSERT INTO senses VALUES(102,12,'The activity of running','A2',NULL,0)"
        )
    await catalog.import_reference(source)
    provider = Transport()
    catalog.llm = provider
    await catalog.start()
    word = await catalog.generate_word("run", example_count=1)
    return word.id, provider


async def test_retry_publishes_a_saved_word_after_index_update_failure(
    catalog, source, monkeypatch
):
    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE words SET word='run',display_form='run'")
        connection.execute("UPDATE entries SET pos='verb'")
        connection.execute("UPDATE senses SET definition='Move quickly on foot'")
    await catalog.import_reference(source)
    provider = Transport()
    catalog.llm = provider
    await catalog.start()
    update = catalog._search.update

    async def failed_update(word_id):
        raise RuntimeError("index update failed")

    monkeypatch.setattr(catalog._search, "update", failed_update)
    with pytest.raises(RuntimeError, match="index update failed"):
        await catalog.generate_word("run", example_count=1)
    assert (await catalog.search("run", include_reference=True)).items[0].kind == "REFERENCE"
    before = list(provider.calls)
    monkeypatch.setattr(catalog._search, "update", update)
    saved = await catalog.generate_word("run", example_count=1)
    assert provider.calls == before
    hit = (await catalog.search("sprint", include_reference=True)).items[0]
    assert hit.kind == "WORD" and hit.word_id == saved.id
    assert all(
        hit.kind == "WORD" or hit.reference_id != encode_reference_id(1)
        for hit in (await catalog.search("run", include_reference=True)).items
    )


async def test_generation_reuses_ranked_forms_aliases_and_patterns(catalog, source):
    word_id, provider = await prepare_run(catalog, source)
    matches = await catalog.search("running", include_reference=True)
    assert [(hit.kind, hit.match_kind) for hit in matches.items[:2]] == [
        ("WORD", MatchKind.EXACT),
        ("REFERENCE", MatchKind.EXACT),
    ]
    assert matches.items[0].word_id == word_id
    assert matches.items[1].display == "running"
    before = list(provider.calls)
    catalog.llm = None
    targets = ["RUN", "running", "sprint", "running the business"]
    reused = await asyncio.gather(
        *(catalog.generate_word(target, with_usage=True) for target in targets)
    )
    assert {word.id for word, _ in reused} == {word_id}
    assert all(usage == [] for _, usage in reused)
    assert provider.calls == before
    assert len((await catalog.search("run", include_reference=True)).items) >= 1
    assert all(
        hit.kind == "WORD" or hit.reference_id != encode_reference_id(1)
        for hit in (await catalog.search("run", include_reference=True)).items
    )


async def test_generation_uses_the_selected_reference_identity_for_an_inflected_input(
    catalog, source
):
    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE words SET word='run',display_form='run'")
        connection.execute("UPDATE entries SET pos='verb'")
        connection.execute("UPDATE senses SET definition='Move quickly on foot'")
        connection.execute(
            "INSERT INTO entry_inflections VALUES(1,11,'present participle','running')"
        )
    await catalog.import_reference(source)
    provider = Transport()
    catalog.llm = provider
    await catalog.start()
    hit = (await catalog.search("running", include_reference=True)).items[0]
    assert (hit.kind, hit.display, hit.matched_surface) == ("REFERENCE", "run", "running")
    for query, kind in [("runn", MatchKind.PREFIX), ("unning", MatchKind.SUBSTRING)]:
        hit = (await catalog.search(query, include_reference=True)).items[0]
        assert (hit.kind, hit.match_kind, hit.matched_surface) == ("REFERENCE", kind, "running")
    result = await catalog.generate_word("running", example_count=1)
    assert result.lemma == "run"
    assert provider.calls == ["InventoryOutput", "EnrichmentBatch"]
    catalog.llm = None
    reused = await catalog.generate_word("running")
    assert reused == result
    for query, kind, surface in [
        ("sprint", MatchKind.EXACT, "sprint"),
        ("spri", MatchKind.PREFIX, "sprint"),
        ("prin", MatchKind.SUBSTRING, "sprint"),
        ("running", MatchKind.EXACT, "running"),
        ("runn", MatchKind.PREFIX, "running"),
        ("unning", MatchKind.SUBSTRING, "running"),
        ("running the business", MatchKind.EXACT, "run {sth}"),
    ]:
        hits = (await catalog.search(query, include_reference=True)).items
        assert all(hit.kind == "WORD" for hit in hits)
        assert (hits[0].word_id, hits[0].match_kind, hits[0].matched_surface) == (
            result.id,
            kind,
            surface,
        )
    assert (await catalog.search("the business", include_reference=True)).items == []


async def test_reused_words_fill_one_missing_theme_for_overlapping_inputs(catalog, source):
    word_id, provider = await prepare_run(catalog, source)
    await catalog.create_theme("sport", "Sport", "Training")
    result = await asyncio.gather(
        *(
            catalog.generate_word(target, theme="sport", example_count=1)
            for target in ["running", "sprint"]
        )
    )
    assert {word.id for word in result} == {word_id}
    assert provider.calls.count("ThemedWord") == 1
    catalog.llm = None
    reused, usage = await catalog.generate_word("running", theme="sport", with_usage=True)
    assert reused.id == word_id and usage == []


async def test_explicit_reference_and_ranked_word_share_theme_generation(catalog, source):
    word_id, provider = await prepare_run(catalog, source)
    await catalog.create_theme("sport", "Sport", "Training")
    entered, release = asyncio.Event(), asyncio.Event()
    original = provider.complete

    async def blocked(instruction, data, schema):
        if schema.__name__ == "ThemedWord":
            entered.set()
            await release.wait()
        return await original(instruction, data, schema)

    provider.complete = blocked
    selected = asyncio.create_task(
        catalog.generate_word(
            "run", reference_id=encode_reference_id(1), theme="sport", example_count=1
        )
    )
    ranked = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        ranked = asyncio.create_task(
            catalog.generate_word("running", theme="sport", example_count=1)
        )
        release.set()
        results = await asyncio.gather(selected, ranked)
        assert {word.id for word in results} == {word_id}
        assert provider.calls.count("ThemedWord") == 1
    finally:
        release.set()
        await asyncio.gather(
            *(task for task in (selected, ranked) if task is not None), return_exceptions=True
        )


async def test_questions_do_not_generate_missing_word_theme(catalog, source):
    word_id, _ = await prepare_run(catalog, source)
    await catalog.create_theme("sport", "Sport", "Training")
    provider = catalog.llm
    before = list(provider.calls)
    sense_id = (await catalog.get_word(word_id)).senses[0].id
    with pytest.raises(InvalidResourceError):
        await catalog.generate_questions(
            sense_id, QuestionType.WORD_TO_DEFINITION, 1, theme="sport", distractor_count=3
        )
    assert provider.calls == before


async def test_generation_does_not_treat_suggestions_as_saved_words(catalog, source):
    await prepare_run(catalog, source)
    catalog.llm = None
    for target, kind in [
        ("runn", MatchKind.PREFIX),
        ("unning", MatchKind.SUBSTRING),
        ("runnng", MatchKind.FUZZY),
    ]:
        first = (await catalog.search(target, include_reference=True)).items[0]
        assert first.match_kind is kind
        with pytest.raises(InvalidResourceError, match="no exact search match"):
            await catalog.generate_word(target)


async def test_unified_ranking_limit_and_reference_replacement(catalog, source):
    with sqlite3.connect(source) as connection:
        connection.execute("INSERT INTO word_alternatives VALUES(1,'banking','inflection')")
        connection.execute("INSERT INTO words VALUES(2,'banker','banker','word','done')")
        connection.execute("INSERT INTO entries VALUES(12,2,'noun',0,NULL,NULL,NULL)")
        connection.execute("INSERT INTO senses VALUES(102,12,'A bank worker','A2',NULL,0)")
    await catalog.import_reference(source)
    async with catalog.db.transaction() as session:
        session.add_all(
            [
                Word(
                    lemma=f"banker {i:02d}",
                    match_key=f"banker {i:02d}",
                    entry_type="WORD",
                    generation_state="DONE",
                )
                for i in range(40)
            ]
        )
    await catalog.start()
    matches = await catalog.search("banker", include_reference=True)
    assert len(matches.items) == 30
    assert matches.items[0].kind == "REFERENCE"
    assert matches.items[0].match_kind is MatchKind.EXACT
    assert all(
        hit.kind == "WORD" and hit.match_kind is MatchKind.PREFIX for hit in matches.items[1:]
    )
    assert len({hit.word_id for hit in matches.items[1:]}) == 29
    result = await catalog._search.search("banker", include_reference=True, limit=1)
    assert len(result.items) == 1 and result.items[0] == matches.items[0]


def test_search_result_has_one_discriminated_list_and_validated_match_kinds():
    adapter = TypeAdapter(SearchResult)
    value = {
        "items": [
            {
                "kind": "WORD",
                "word_id": 1,
                "lemma": "run",
                "entry_type": "WORD",
                "match_kind": "EXACT",
                "matched_surface": "running",
            },
            {
                "kind": "REFERENCE",
                "reference_id": "opaque",
                "display": "runner",
                "entry_type": "WORD",
                "match_kind": "PREFIX",
                "matched_surface": "runner",
            },
        ]
    }
    parsed = adapter.validate_python(value)
    assert adapter.dump_python(parsed, mode="json") == value
    assert not hasattr(parsed, "words") and not hasattr(parsed, "references")
    value["items"][0]["match_kind"] = "FORM"
    with pytest.raises(ValidationError):
        adapter.validate_python(value)


@pytest.mark.parametrize(
    ("query", "kind"),
    [
        ("running", MatchKind.EXACT),
        ("runn", MatchKind.PREFIX),
        ("unning", MatchKind.SUBSTRING),
        ("runnng", MatchKind.FUZZY),
    ],
)
async def test_every_match_class_prefers_lemma_then_alias_then_form(catalog, source, query, kind):
    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE words SET word='running', display_form='running'")
    await catalog.import_reference(source)
    async with catalog.db.transaction() as session:
        session.add_all(
            [
                Word(id=i, lemma=name, match_key=name, entry_type="WORD", generation_state="DONE")
                for i, name in [(1, "running"), (2, "z-alias"), (3, "a-form"), (4, "b-alias")]
            ]
        )
        await session.flush()
        session.add_all(
            [WordAlias(word_id=i, content="running", match_key="running") for i in (2, 4)]
        )
        sense = Sense(word_id=3, pos="VERB", tier="CORE")
        session.add(sense)
        await session.flush()
        session.add(SenseForm(sense_id=sense.id, surface="running", inf="ING"))
    await catalog.start()
    hits = (await catalog.search(query, include_reference=True)).items
    assert [(hit.word_id, hit.match_kind, hit.matched_surface) for hit in hits[:-1]] == [
        (i, kind, "running") for i in (1, 4, 2, 3)
    ]
    assert (hits[-1].kind, hits[-1].match_kind, hits[-1].matched_surface) == (
        "REFERENCE",
        kind,
        "running",
    )


@pytest.mark.parametrize("query", ["running", "runn", "unning", "runnng"])
def test_reference_surface_origins_rank_for_all_match_classes(query):
    snapshot = Search._build(
        [],
        [],
        [],
        [],
        set(),
        [
            (SourceHit(1, "a-owner", "word"), ["a-owner", "a-owner", "running"], []),
            (SourceHit(2, "running", "word"), ["running", "running"], []),
            (SourceHit(3, "0-owner", "word"), ["0-owner", "0-owner"], ["running"]),
        ],
    )[2]
    hits = Search._find(snapshot, query, "CAMBRIDGE", 30)
    assert [hit.reference_id for _, hit in hits] == [encode_reference_id(i) for i in (2, 1, 3)]
    assert all(hit.matched_surface == "running" for _, hit in hits)


@pytest.mark.parametrize("query", ["running", "runn", "unning", "runnng"])
def test_sorting_precedes_the_limit_and_deduplicates_surfaces(query):
    words = [(i, f"owner {40 - i:02d}", "WORD") for i in range(1, 41)]
    aliases = [(i, surface) for i in range(1, 41) for surface in ("running", "running again")]
    snapshot = Search._build(words, aliases, [], [], set(), [])[2]
    hits = Search._find(snapshot, query, "LEXI", 30)
    assert [hit.lemma for _, hit in hits] == [f"owner {i:02d}" for i in range(30)]
    assert len({hit.word_id for _, hit in hits}) == 30
    assert Search._find(snapshot, query, "LEXI", 1)[0][1] == hits[0][1]


def test_fuzzy_evidence_is_a_surface_matched_by_the_engine():
    snapshot = Search._build(
        [(1, "owner", "WORD")], [(1, "abcxyf"), (1, "abcxxxdef")], [], [], set(), []
    )[2]
    hit = Search._find(snapshot, "abcdef", "LEXI", 30)[0][1]
    assert hit.match_kind is MatchKind.FUZZY
    assert hit.matched_surface == "abcxyf"


@pytest.mark.parametrize("query", ["running", "runn", "unning", "runnng"])
def test_raw_surface_scores_tie_and_labels_then_ids_decide_order(query):
    snapshot = Search._build(
        [(1, "a-owner", "WORD"), (2, "z-owner", "WORD")],
        [(1, "running"), (1, "other"), (1, "extra"), (2, "running")],
        [],
        [],
        set(),
        [(SourceHit(i, "running", "word"), ["running", "running"], []) for i in (40, 2)],
    )[2]
    index, _ = snapshot
    schema, searcher = index.schema, index.searcher()
    if query == "running":
        branch = tantivy.Query.term_query(schema, "surface", query)
    elif query == "runn":
        branch = tantivy.Query.regex_query(schema, "surface", query + ".*")
    elif query == "unning":
        branch = tantivy.Query.regex_query(schema, "surface", ".*" + query + ".*")
    else:
        branch = tantivy.Query.fuzzy_term_query(schema, "surface", query, distance=2)
    combined = tantivy.Query.boolean_query(
        [
            (tantivy.Occur.Must, tantivy.Query.term_query(schema, "source", "LEXI")),
            (tantivy.Occur.Must, branch),
        ]
    )
    engine_scores = {score for score, _ in searcher.search(combined, limit=20).hits}
    assert len(engine_scores) == 1
    results = Search._find(snapshot, query, "LEXI", 30)
    assert {-order[3] for order, _ in results} == {1.0}
    assert [hit.word_id for _, hit in results] == [1, 2]
    assert [hit.reference_id for _, hit in Search._find(snapshot, query, "CAMBRIDGE", 1)] == [
        encode_reference_id(2)
    ]


def test_multiple_matching_surfaces_do_not_exhaust_the_identity_limit():
    words = [(i, f"owner {i:02d}", "WORD") for i in range(40)]
    aliases = [(i, f"running {j:03d}") for i in range(40) for j in range(50 if i == 0 else 1)]
    snapshot = Search._build(words, aliases, [], [], set(), [])[2]
    hits = Search._find(snapshot, "runn", "LEXI", 30)
    assert [hit.word_id for _, hit in hits] == list(range(30))


def test_short_inputs_still_match_substrings_and_anchored_patterns_but_not_fuzzy():
    snapshot = Search._build([(1, "running", "WORD")], [], [], [(1, 1, "a{sth}", None)], set(), [])[
        2
    ]
    for query in ("un", "n"):
        hit = Search._find(snapshot, query, "LEXI", 30)[0][1]
        assert (hit.match_kind, hit.matched_surface) == (MatchKind.SUBSTRING, "running")
    assert Search._find(snapshot, "rn", "LEXI", 30) == []
    hit = Search._find(snapshot, "ab", "LEXI", 30)[0][1]
    assert (hit.match_kind, hit.matched_surface) == (MatchKind.EXACT, "a{sth}")
