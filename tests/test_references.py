import asyncio
import hashlib
import sqlite3
import time
from types import SimpleNamespace

import pytest

from lexi_ai.errors import InvalidHandleError
from lexi_ai.references.cambridge import Cambridge, decode_reference_id, encode_reference_id
from lexi_ai.references.wordnet import Synset, lookup


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "reference.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            "CREATE TABLE words(id INTEGER PRIMARY KEY, word TEXT, display_form TEXT, "
            "entry_type TEXT, status TEXT);"
            "CREATE TABLE entries(id INTEGER PRIMARY KEY, word_id INTEGER, pos TEXT, "
            "entry_order INTEGER, pronunciation_uk TEXT, pronunciation_us TEXT, headword TEXT);"
            "CREATE TABLE senses(id INTEGER PRIMARY KEY, entry_id INTEGER, definition TEXT, "
            "cefr_level TEXT, phrase_title TEXT, sense_order INTEGER);"
            "CREATE TABLE examples(id INTEGER PRIMARY KEY, sense_id INTEGER, example TEXT, "
            "example_order INTEGER);"
            "CREATE TABLE word_alternatives(word_id INTEGER, alternative_word TEXT, "
            "alternative_type TEXT);"
            "CREATE TABLE entry_inflections(id INTEGER PRIMARY KEY, entry_id INTEGER, "
            "form_type TEXT, inflected_form TEXT);"
            "INSERT INTO words VALUES(1,'bank','bank','word','done');"
            "INSERT INTO words VALUES(2,'bank','bank','word','done');"
            "INSERT INTO words VALUES(3,'empty','empty','word','done');"
            "INSERT INTO words VALUES(4,'pending','pending','word','pending');"
            "INSERT INTO entries VALUES(11,1,'noun',0,NULL,NULL,'bank');"
            "INSERT INTO entries VALUES(12,2,'verb',0,NULL,NULL,'bank');"
            "INSERT INTO senses VALUES(101,11,'financial institution','B1',NULL,0);"
            "INSERT INTO senses VALUES(102,12,'tilt an aircraft','C1',NULL,0);"
            "INSERT INTO examples VALUES(1,101,'Go to the bank.',0);"
        )
    return path


async def test_fetch_by_id_is_exact_and_read_only(source):
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    reference = Cambridge(source)
    assert (await reference.fetch_by_id(1)).senses[0].definition == "financial institution"
    assert (await reference.fetch_by_id(1)).senses[0].headword == "bank"
    assert (await reference.fetch_by_id(2)).senses[0].pos == "verb"
    assert (await reference.fetch_by_id(500)) is None
    assert (await reference.fetch_by_id(1)).senses[0].examples == ["Go to the bank."]
    assert (await reference.fetch_by_id(3)).senses == []
    assert decode_reference_id(encode_reference_id(1)) == 1
    with pytest.raises(InvalidHandleError):
        decode_reference_id("entry_!!!")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


async def test_projection_preserves_distinct_eligible_ids(source):
    with sqlite3.connect(source) as connection:
        connection.execute("INSERT INTO entry_inflections VALUES(1,11,'past tense','banked')")
    found = await Cambridge(source).projection()
    assert [item.id for item, _, _ in found] == [1, 2]
    assert all(surfaces == ["bank", "bank"] for _, surfaces, _ in found)
    assert found[0][2] == ["banked"]
    assert found[1][2] == []
    assert await lookup("not/a/citation") == []


async def test_wordnet_workers_serialize_lookup_and_materialize_under_same_lock(monkeypatch):
    import nltk.corpus

    active = 0
    calls = []

    def synsets(citation):
        nonlocal active
        assert active == 0
        active += 1
        calls.append(citation)
        time.sleep(0.005)

        def examples():
            nonlocal active
            assert active == 1
            time.sleep(0.005)
            active -= 1
            return ["Example"]

        return [
            SimpleNamespace(
                name=lambda: citation + ".n.01",
                pos=lambda: "n",
                definition=lambda: "Meaning",
                examples=examples,
            )
        ]

    monkeypatch.setattr(nltk.corpus, "wordnet", SimpleNamespace(synsets=synsets))
    citations = [f"word {index}" for index in range(8)]
    results = await asyncio.gather(*(lookup(citation) for citation in citations))
    assert active == 0
    assert set(calls) == {citation.replace(" ", "_") for citation in citations}
    assert results == [
        [Synset(citation.replace(" ", "_") + ".n.01", "n", "Meaning", ["Example"])]
        for citation in citations
    ]


async def test_wordnet_missing_corpus_releases_lock_for_following_lookups(monkeypatch):
    import nltk.corpus

    calls = []

    def synsets(citation):
        calls.append(citation)
        raise LookupError("missing corpus")

    monkeypatch.setattr(nltk.corpus, "wordnet", SimpleNamespace(synsets=synsets))
    assert await asyncio.gather(lookup("bank"), lookup("charge")) == [[], []]
    assert set(calls) == {"bank", "charge"}


async def test_reference_fetch_aggregates_examples_once_with_stable_order(source, monkeypatch):
    with sqlite3.connect(source) as connection:
        connection.executemany(
            "INSERT INTO senses VALUES(?,11,?,'A2',NULL,?)",
            [(200 + i, f"meaning{i}", i + 1) for i in range(95)],
        )
        connection.executemany(
            "INSERT INTO examples VALUES(?,?,?,?)",
            [
                (2000 + i * 3 + j, 200 + i, f"example{i}-{j}", 2 - j)
                for i in range(95)
                for j in range(3)
            ],
        )
        connection.execute("INSERT INTO word_alternatives VALUES(1,'banks','plural')")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    reference = Cambridge(source)
    connect = reference._connect
    statements = []

    def traced():
        connection = connect()
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(reference, "_connect", traced)
    entry = await reference.fetch_by_id(1)
    assert len(entry.senses) == 96
    assert len(statements) == 1
    assert entry.senses[1].examples == ["example0-2", "example0-1", "example0-0"]
    assert entry.senses[-1].definition == "meaning94"
    assert entry.alternatives == [("banks", "plural")]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


async def test_reference_preserves_affix_and_phrase_identity(source):
    with sqlite3.connect(source) as connection:
        connection.execute("INSERT INTO entries VALUES(13,1,'prefix',1,NULL,NULL,'bank-')")
        connection.execute("INSERT INTO senses VALUES(103,13,'Affix meaning',NULL,NULL,0)")
        connection.execute("UPDATE senses SET phrase_title='bank on' WHERE id=101")
    entry = await Cambridge(source).fetch_by_id(1)
    assert entry.senses[0].phrase_title == "bank on"
    assert entry.senses[1].headword == "bank-"
    assert entry.senses[1].pos == "prefix"
