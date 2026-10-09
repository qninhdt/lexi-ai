"""Real SQLite artifacts, PostgreSQL imports and isolated download transport checks."""

import asyncio
import hashlib
import io
import os
import sqlite3
from contextlib import closing
from uuid import uuid4

import pytest
from sqlalchemy import select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from lexi_ai import Lexicon, migrations
from lexi_ai import datasets as distribution
from lexi_ai.errors import InvalidHandleError, InvalidResourceError
from lexi_ai.models import Option, Question
from lexi_ai.questions.storage import append
from lexi_ai.references.cambridge import Cambridge, encode_reference_id
from lexi_ai.references.schema import datasets
from lexi_ai.schema import Definition, Sense, Word


def test_download_validates_and_reuses_cached_artifact(tmp_path, monkeypatch):
    payload = b"reference artifact"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(distribution, "REFERENCE_SHA256", hashlib.sha256(payload).hexdigest())
    calls = []

    def fetch(url, timeout):
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(distribution, "urlopen", fetch)
    path = distribution.download("reference")
    assert path.name == "reference.sqlite" and path.read_bytes() == payload
    assert distribution.download("reference") == path
    assert calls == [distribution.REFERENCE_URL]
    path.write_bytes(b"corrupted cache")
    assert distribution.download("reference").read_bytes() == payload
    assert len(calls) == 2
    assert not list(tmp_path.rglob("*.download"))


def test_bad_download_does_not_publish_partial_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(distribution, "urlopen", lambda *a, **k: io.BytesIO(b"incorrect"))
    with pytest.raises(InvalidResourceError, match="checksum mismatch"):
        distribution.download("reference")
    assert not list(tmp_path.rglob("*.sqlite"))
    assert not list(tmp_path.rglob("*.download"))


@pytest.fixture
async def dictionary():
    target = os.getenv("LEXI_TEST_PG_URL")
    if not target:
        pytest.skip("no disposable LEXI_TEST_PG_URL")
    admin_url = make_url(target)
    admin = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    name = "lexi_dataset_test_" + uuid4().hex
    lexicon = None
    try:
        async with admin.connect() as connection:
            await connection.execute(text(f'CREATE DATABASE "{name}"'))
        url = admin_url.set(database=name).render_as_string(hide_password=False)
        await asyncio.to_thread(migrations.upgrade_to_head, url)
        lexicon = Lexicon(url)
        yield lexicon
    finally:
        if lexicon:
            await lexicon.close()
        async with admin.connect() as connection:
            await connection.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        await admin.dispose()


async def test_reference_import_preserves_evidence_and_needs_no_file_at_runtime(
    dictionary, source, monkeypatch
):
    with sqlite3.connect(source) as connection:
        connection.execute("INSERT INTO entry_inflections VALUES(1,11,'plural','banks')")
    expected = await Cambridge(source).fetch_by_id(1)
    assert await dictionary.import_reference(source) == 4
    assert await dictionary.import_reference(source) == 0

    def no_download(*args):
        pytest.fail("an installed reference must not require network or cache")

    monkeypatch.setattr(distribution, "download", no_download)
    assert await dictionary.import_reference() == 0
    source.unlink()
    assert await dictionary._cambridge.fetch_by_id(1) == expected
    await dictionary.start()
    result = await dictionary.search("bank", include_reference=True)
    assert result.items[0].reference_id == encode_reference_id(1)
    inflected = (await dictionary.search("banks", include_reference=True)).items[0]
    assert (inflected.display, str(inflected.match_kind), inflected.matched_surface) == (
        "bank",
        "EXACT",
        "banks",
    )
    await dictionary.validate_reference(result.items[0].reference_id)
    with pytest.raises(InvalidHandleError):
        await dictionary.validate_reference(encode_reference_id(500))
    assert dictionary.llm is None


async def test_reference_conflict_rolls_back_all_tables_and_marker(dictionary, source):
    await dictionary.import_reference(source)
    async with dictionary.db.read() as connection:
        original = await connection.scalar(select(datasets.c.checksum))
    with sqlite3.connect(source) as connection:
        connection.execute("INSERT INTO words VALUES(2,'new','new','word','done')")
        connection.execute("UPDATE senses SET definition='changed meaning' WHERE id=101")
    with pytest.raises(InvalidResourceError, match="conflict"):
        await dictionary.import_reference(source)
    assert await dictionary._cambridge.fetch_by_id(2) is None
    assert (await dictionary._cambridge.fetch_by_id(1)).senses[
        0
    ].definition == "Financial institution"
    async with dictionary.db.read() as connection:
        assert await connection.scalar(select(datasets.c.checksum)) == original


@pytest.fixture
async def content(tmp_path):
    path = tmp_path / "content.sqlite"
    url = "sqlite+aiosqlite:///" + str(path)
    await asyncio.to_thread(migrations.upgrade_to_head, url)
    lexicon = Lexicon(url)
    try:
        async with lexicon.db.transaction() as session:
            session.add(
                Word(
                    id=40,
                    lemma="bank",
                    match_key="bank",
                    entry_type="WORD",
                    generation_state="DONE",
                )
            )
            await session.flush()
            session.add(Sense(id=70, word_id=40, pos="NOUN", tier="CORE"))
            await session.flush()
            session.add(Definition(id=80, sense_id=70, content="a financial institution"))
        question = (
            await append(
                lexicon.db,
                [
                    Question(
                        0,
                        70,
                        None,
                        "DEFINITION_TO_WORD",
                        "a financial institution",
                        Option("correct", "bank", "matches"),
                        [Option("wrong", "river", "different")],
                    )
                ],
            )
        )[0]
    finally:
        await lexicon.close()
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode=DELETE")
    return path, question


async def test_content_words_then_questions_preserves_ids_and_advances_sequences(
    dictionary, source, content, monkeypatch
):
    await dictionary.import_reference(source)
    path, question = content
    assert await dictionary.import_content(path, questions=False) == 3
    assert (await dictionary.get_word(40)).senses[0].id == 70
    assert await dictionary.get_question(question.id) is None
    assert await dictionary.import_content(path) == 1
    assert await dictionary.get_question(question.id) == question
    assert await dictionary.import_content(path) == 0

    def no_download(*args):
        pytest.fail("installed content must not download again")

    monkeypatch.setattr(distribution, "download", no_download)
    assert await dictionary.import_content() == 0
    async with dictionary.db.transaction() as session:
        word = Word(lemma="river", match_key="river", entry_type="WORD", generation_state="DONE")
        session.add(word)
        await session.flush()
        assert word.id > 40


async def test_content_conflict_preserves_existing_senses(dictionary, source, content):
    await dictionary.import_reference(source)
    path, _ = content
    await dictionary.import_content(path, questions=False)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE definitions SET content='different' WHERE id=80")
    with pytest.raises(InvalidResourceError, match="conflict"):
        await dictionary.import_content(path)
    assert (await dictionary.get_word(40)).senses[0].definition.content == "a financial institution"
    async with dictionary.db.read() as connection:
        assert (
            await connection.scalar(
                select(datasets.c.name).where(datasets.c.name == "content.lexi.questions")
            )
            is None
        )


async def test_content_requires_reference(dictionary, content):
    path, _ = content
    with pytest.raises(InvalidResourceError, match="import reference"):
        await dictionary.import_content(path)
    assert await dictionary.get_word(40) is None


async def test_content_import_is_tracked_per_destination_schema(dictionary, source, content):
    await dictionary.import_reference(source)
    path, question = content
    assert await dictionary.import_content(path) == 4
    url = dictionary.db.engine.url.render_as_string(hide_password=False)
    schema = "second_catalog"
    await asyncio.to_thread(migrations.upgrade_to_head, url, db_schema=schema)
    second = Lexicon(url, db_schema=schema)
    try:
        assert await second.import_content(path) == 4
        assert await second.get_word(40) == await dictionary.get_word(40)
        assert await second.get_question(question.id) == question
        assert await second.import_content() == 0
        assert await dictionary.import_content() == 0
        async with second.db.transaction() as session:
            word = Word(
                lemma="river", match_key="river", entry_type="WORD", generation_state="DONE"
            )
            session.add(word)
            await session.flush()
            assert word.id > 40
    finally:
        await second.close()


async def test_invalid_question_payload_rolls_back_content(dictionary, source, content):
    await dictionary.import_reference(source)
    path, _ = content
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE questions SET payload='{}'")
    with pytest.raises(InvalidResourceError, match="invalid question payload"):
        await dictionary.import_content(path)
    assert await dictionary.get_word(40) is None
