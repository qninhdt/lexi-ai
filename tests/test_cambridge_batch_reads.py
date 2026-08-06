"""`first_definitions` answers a batch in one query, not one per word.

This is a read on the search path: `search()` asks for a gloss per candidate to
disambiguate them, so the cost used to scale with the number of results. Ten
candidates meant ten round trips against the reference database, each opening its
own statement on a connection opened for the call.

Two things are asserted, and both matter. Equivalence, because a rewritten query
that returns different definitions is worse than a slow one — the gloss feeds
disambiguation, so a wrong first sense mislabels a word. And the query count,
because equivalence alone would pass for the loop this replaced.
"""

import sqlite3

import pytest

from lexi_ai.references.cambridge import CambridgeSource

pytestmark = pytest.mark.asyncio


@pytest.fixture
def multi_word_cambridge(tmp_path):
    """Several words, each with more than one sense across more than one entry.

    The ordering is the point. `first_definitions` promises the first sense by
    `(entry_order, sense_order)`, so every word here has its senses inserted out
    of that order — a query that returned whichever row the plan reached first
    would pass against a fixture where insertion order already matched.
    """
    db_path = tmp_path / "cam-multi.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE words (
            id INTEGER PRIMARY KEY, word TEXT, display_form TEXT,
            entry_type TEXT, status TEXT
        );
        CREATE TABLE entries (
            id INTEGER PRIMARY KEY, word_id INTEGER, entry_order INTEGER, pos TEXT
        );
        CREATE TABLE senses (
            id INTEGER PRIMARY KEY, entry_id INTEGER, sense_order INTEGER,
            guideword TEXT, definition TEXT, cefr_level TEXT, domain TEXT,
            phrase_title TEXT
        );
        """
    )
    conn.executescript(
        """
        INSERT INTO words VALUES
            (1,'bank','bank','word','done'),
            (2,'spring','spring','word','done'),
            (3,'light','light','word','done'),
            (4,'silent','silent','word','done');
        -- `bank`: two entries, inserted with the later one first.
        INSERT INTO entries VALUES (20,1,1,'verb'), (10,1,0,'noun');
        INSERT INTO senses VALUES
            (200,20,0,'','to put money in an account','B1',NULL,NULL),
            (100,10,1,'','the side of a river','B2',NULL,NULL),
            (101,10,0,'','a place that keeps money','A2',NULL,NULL);
        -- `spring`: one entry, senses inserted in reverse order.
        INSERT INTO entries VALUES (30,2,0,'noun');
        INSERT INTO senses VALUES
            (300,30,2,'','a coiled piece of metal','B2',NULL,NULL),
            (301,30,0,'','the season after winter','A1',NULL,NULL);
        -- `light`: a first sense with an empty definition, which is skipped
        -- rather than returned as a blank gloss.
        INSERT INTO entries VALUES (40,3,0,'noun');
        INSERT INTO senses VALUES (400,40,0,'','','A1',NULL,NULL);
        -- `silent` has an entry but no senses at all.
        INSERT INTO entries VALUES (50,4,0,'adjective');
        """
    )
    conn.commit()
    conn.close()
    return str(db_path)


async def test_first_definitions_returns_each_word_s_first_sense(multi_word_cambridge):
    source = CambridgeSource(multi_word_cambridge)

    found = await source.first_definitions([1, 2, 3, 4, 999])

    # Ordered by (entry_order, sense_order), so entry 10 beats entry 20 and
    # sense_order 0 beats 1 — neither of which is insertion order here.
    assert found[1] == "a place that keeps money"
    assert found[2] == "the season after winter"
    # An empty definition is absent rather than present-and-blank: the caller uses
    # this as a gloss, and a blank one reads as "this sense has no meaning".
    assert 3 not in found
    # A word with no senses, and an id that does not exist at all.
    assert 4 not in found
    assert 999 not in found


async def test_first_definitions_costs_one_query_for_the_whole_batch(
    multi_word_cambridge, monkeypatch
):
    """The bound this rewrite exists for.

    Counted at the connection rather than timed: a timing assertion on a local
    SQLite file would be noise, while the statement count is exactly the thing
    that used to grow with the batch.
    """
    source = CambridgeSource(multi_word_cambridge)

    executed: list[str] = []
    real_connect = source._connect

    def counting_connect():
        # `set_trace_callback` is SQLite's own hook and fires for every statement
        # the connection runs. Wrapping `execute` is not an option: it is a
        # read-only attribute on `sqlite3.Connection`.
        connection = real_connect()
        connection.set_trace_callback(executed.append)
        return connection

    monkeypatch.setattr(source, "_connect", counting_connect)

    await source.first_definitions([1, 2, 3, 4])

    assert len(executed) == 1, (
        f"{len(executed)} statements for a 4-word batch, so the read is still one "
        "query per word and its cost scales with the result count"
    )


async def test_first_definitions_asks_nothing_of_an_empty_batch(multi_word_cambridge):
    """No ids means no connection, rather than a query with an empty IN list."""
    source = CambridgeSource(multi_word_cambridge)

    def refuse():  # pragma: no cover - asserted by not being called
        raise AssertionError("opened a connection for an empty batch")

    source._connect = refuse  # type: ignore[method-assign]

    assert await source.first_definitions([]) == {}
