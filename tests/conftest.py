"""Disposable Reference snapshots and PostgreSQL schema-isolated test dictionaries."""

import os
import sqlite3
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from lexi_ai.db.session import Database
from lexi_ai.schema import Base


def pytest_sessionstart(session):
    if os.getenv("LEXI_REQUIRE_PG") == "1" and not os.getenv("LEXI_TEST_PG_URL"):
        raise pytest.UsageError(
            "LEXI_REQUIRE_PG=1 requires an explicit disposable LEXI_TEST_PG_URL"
        )


@pytest.fixture
async def pg_db():
    url = os.getenv("LEXI_TEST_PG_URL")
    if not url:
        pytest.skip("no disposable LEXI_TEST_PG_URL")
    schema = f"lexi_search_test_{uuid.uuid4().hex[:12]}"
    engine = create_async_engine(url)
    database = None
    try:
        async with engine.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        database = Database(url, schema=schema)
        await database.create_schema(Base.metadata)
        yield database
    finally:
        if database is not None:
            await database.close()
        async with engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "reference.db"
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
            "INSERT INTO entries VALUES(11,1,'noun',0,NULL,NULL,'bank');"
            "INSERT INTO senses VALUES(101,11,'Financial institution','A2',NULL,0);"
        )
    return path
