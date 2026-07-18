"""Tests for the reference loader (Phase 3).

Run against the real Cambridge ``./data`` SQLite file and the installed WordNet
corpus. Skipped automatically if ``./data`` is absent.
"""

import os
import sqlite3

import pytest

from lexi_ai.references.cambridge import CambridgeSource
from lexi_ai.references.loader import ReferenceLoader
from lexi_ai.references.wordnet import WordNetSource

CAMBRIDGE_PATH = os.environ.get("LEXI_CAMBRIDGE_DB_PATH", "./data")

pytestmark = pytest.mark.skipif(
    not os.path.exists(CAMBRIDGE_PATH),
    reason="Cambridge ./data not present",
)


@pytest.fixture
def loader():
    return ReferenceLoader(CambridgeSource(CAMBRIDGE_PATH), WordNetSource())


async def test_bundle_book_has_both_sources(loader):
    bundle = await loader.bundle("book")
    assert bundle.has_cambridge
    assert bundle.cambridge_word_id is not None
    assert len(bundle.cambridge_senses) > 0
    # book has strong WordNet coverage.
    assert bundle.has_wordnet
    # Sanity: a sense carries a definition and at least some examples exist.
    assert all(s.definition for s in bundle.cambridge_senses)


async def test_bundle_book_cefr_present(loader):
    bundle = await loader.bundle("book")
    cefr_values = {s.cefr_level for s in bundle.cambridge_senses if s.cefr_level}
    assert cefr_values, "expected at least one Cambridge cefr_level for 'book'"


async def test_bundle_expression_empty_wordnet_no_crash(loader):
    # An expression with no WordNet synset must not crash; WordNet may be empty.
    bundle = await loader.bundle("act on behalf of")
    # Whether or not Cambridge has it, the call must return a bundle.
    assert bundle is not None
    # WordNet has no 'act_on_behalf_of' synset.
    assert bundle.wordnet_synsets == [] or not bundle.has_cambridge


async def test_wordnet_direct_lookup_counts():
    wn = WordNetSource()
    assert len(await wn.lookup("book")) == 15
    assert len(await wn.lookup("make up")) == 9
    assert await wn.lookup("act on behalf of") == []


async def test_cambridge_read_only(loader):
    # Attempting to write through a mode=ro connection must raise.
    src = CambridgeSource(CAMBRIDGE_PATH)
    assert "mode=ro&immutable=1" in src._uri
    conn = sqlite3.connect(src._uri, uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE _should_fail (x INTEGER)")
    finally:
        conn.close()


async def test_candidates_excludes_done_keys(loader):
    src = CambridgeSource(CAMBRIDGE_PATH)
    from lexi_ai.normalize import match_key

    # Pretend "book" is already generated; it must not appear as a candidate.
    done = {match_key("book")}
    seen = []
    count = 0
    async for surface, _etype in src.candidates(done):
        seen.append(match_key(surface))
        count += 1
        if count >= 500:
            break
    assert match_key("book") not in seen


async def test_wordnet_pos_mapping():
    wn = WordNetSource()
    noun_only = await wn.lookup("book", pos="noun")
    assert noun_only
    assert all(s.pos == "noun" for s in noun_only)
