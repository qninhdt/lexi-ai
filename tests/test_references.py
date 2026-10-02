import asyncio
import hashlib
import sqlite3
import time
from types import SimpleNamespace

import pytest

from lexi_ai.errors import InvalidHandleError
from lexi_ai.references.cambridge import Cambridge, decode_available_id, encode_available_id
from lexi_ai.references.wordnet import Synset, lookup


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "cambridge.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            "CREATE TABLE words(id INTEGER PRIMARY KEY, word TEXT, display_form TEXT, "
            "entry_type TEXT, status TEXT);"
            "CREATE TABLE entries(id INTEGER PRIMARY KEY, word_id INTEGER, pos TEXT, "
            "entry_order INTEGER, pronunciation_uk TEXT, pronunciation_us TEXT);"
            "CREATE TABLE senses(id INTEGER PRIMARY KEY, entry_id INTEGER, definition TEXT, "
            "cefr_level TEXT, phrase_title TEXT, sense_order INTEGER);"
            "CREATE TABLE examples(id INTEGER PRIMARY KEY, sense_id INTEGER, example TEXT, "
            "example_order INTEGER);"
            "CREATE TABLE word_alternatives(word_id INTEGER, alternative_word TEXT, "
            "alternative_type TEXT);"
            "INSERT INTO words VALUES(1,'bank','bank','word','done');"
            "INSERT INTO words VALUES(2,'bank','bank','word','done');"
            "INSERT INTO words VALUES(3,'empty','empty','word','done');"
            "INSERT INTO words VALUES(4,'pending','pending','word','pending');"
            "INSERT INTO entries VALUES(11,1,'noun',0,NULL,NULL);"
            "INSERT INTO entries VALUES(12,2,'verb',0,NULL,NULL);"
            "INSERT INTO senses VALUES(101,11,'financial institution','B1',NULL,0);"
            "INSERT INTO senses VALUES(102,12,'tilt an aircraft','C1',NULL,0);"
            "INSERT INTO examples VALUES(1,101,'Go to the bank.',0);"
        )
    return path


async def test_fetch_by_id_is_exact_and_read_only(source):
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    cambridge = Cambridge(source)
    assert (await cambridge.fetch_by_id(1)).senses[0].definition == "financial institution"
    assert (await cambridge.fetch_by_id(2)).senses[0].pos == "verb"
    assert (await cambridge.fetch_by_id(500)) is None
    assert (await cambridge.fetch_by_id(1)).senses[0].examples == ["Go to the bank."]
    assert (await cambridge.fetch_by_id(3)).senses == []
    assert decode_available_id(encode_available_id(1)) == 1
    with pytest.raises(InvalidHandleError):
        decode_available_id("entry_!!!")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


async def test_search_preserves_distinct_eligible_ids(source):
    found = await Cambridge(source).search("bank")
    assert [item.id for item in found] == [1, 2]
    assert await Cambridge(source).search("empty") == []
    assert await Cambridge(source).search("pending") == []
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


async def test_cambridge_fetch_aggregates_examples_once_with_stable_order(source, monkeypatch):
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
    cambridge = Cambridge(source)
    connect = cambridge._connect
    statements = []

    def traced():
        connection = connect()
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(cambridge, "_connect", traced)
    entry = await cambridge.fetch_by_id(1)
    assert len(entry.senses) == 96
    assert len(statements) == 1
    assert entry.senses[1].examples == ["example0-2", "example0-1", "example0-0"]
    assert entry.senses[-1].definition == "meaning94"
    assert entry.alternatives == [("banks", "plural")]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
