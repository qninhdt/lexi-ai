"""Opt-in dual-DB regression tier — real Postgres over ``LEXI_TEST_PG_URL``.

These tests prove the defect class the SQLite suite cannot see: NUL rejection,
``VARCHAR(n)`` length enforcement, and tz-aware datetime binding on
``TIMESTAMP WITHOUT TIME ZONE``. Each dual-DB regression is written on Phase 0's
harness so it can be confirmed RED on pre-fix code and green after.

Skips cleanly with no driver / no URL — the default ``uv run pytest`` stays
hermetic. The ``pg_session_factory`` fixture is auto-discovered from
``conftest.py``; see it for the install + run invocation.
"""

from sqlalchemy import select

from lexi_ai.db import session_scope
from lexi_ai.models import Word
from lexi_ai.normalize import match_key
from tests.conftest import requires_postgres

pytestmark = requires_postgres


async def test_harness_connects_and_roundtrips_a_word(pg_session_factory):
    """The fixture connects, ``init_models`` builds the schema on Postgres, and a
    trivial ``Word`` round-trips — proving the harness itself before any
    regression leans on it."""
    async with session_scope(pg_session_factory) as session:
        session.add(Word(norm="book", match_key=match_key("book"), status="done"))

    async with session_scope(pg_session_factory) as session:
        row = (await session.execute(select(Word).where(Word.norm == "book"))).scalar_one()
        assert row.match_key == "book"
        assert row.status == "done"
