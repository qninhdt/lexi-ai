"""Tests for style themes (Phase 1: foundation).

Covers the ``theme_key`` normalizer (dedup folding, NO singularization, NUL
safety, empty), the ``Theme`` schema compile, and repository/API resolve-or-
create (dedup, first-seen wins, empty-key raises, list). In-memory SQLite +
StaticPool, mirroring ``test_tags.py``.
"""

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from lexi_ai.api import Lexicon
from lexi_ai.db import create_session_factory, init_models
from lexi_ai.normalize import theme_key
from lexi_ai.persistence.repository import Repository

# --- theme_key normalizer (pure) -----------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Harry Potter", "harry potter"),
        ("Harry Potter", " harry  potter "),
        ("Hárry Potter", "harry potter"),
        ("HARRY POTTER", "harry potter"),
    ],
)
def test_theme_key_folds_variants(a, b):
    assert theme_key(a) == theme_key(b)


def test_theme_key_does_not_singularize():
    # Unlike tag_key, a theme name is a proper voice: "witches" must NOT fold to
    # "witch" (that would corrupt the name and merge two distinct themes).
    assert theme_key("The Witches") == "the witches"
    assert theme_key("Cars") == "cars"


def test_theme_key_strips_control_chars():
    assert theme_key("harry\tpotter") == "harry potter"
    assert "\x00" not in theme_key("har\x00ry")


@pytest.mark.parametrize("blank", ["", "   ", "\t\n", "\x00"])
def test_theme_key_empty_on_blank(blank):
    assert theme_key(blank) == ""


# --- schema compiles on both dialects (extends the models guarantee) ------


def test_theme_tables_compile_on_both_dialects():
    from sqlalchemy.dialects import postgresql, sqlite
    from sqlalchemy.schema import CreateTable

    from lexi_ai.infrastructure.db.models import Base

    names = set(Base.metadata.tables)
    assert {"themes", "themed_senses", "themed_examples"} <= names

    for dialect in (postgresql.dialect(), sqlite.dialect()):
        for table in Base.metadata.sorted_tables:
            ddl = str(CreateTable(table).compile(dialect=dialect))
            assert "CREATE TABLE" in ddl


# --- repository / API resolve-or-create -----------------------------------


@pytest.fixture
async def session_factory():
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
    yield create_session_factory(engine)
    await engine.dispose()


@pytest.fixture
def repo(session_factory):
    return Repository(session_factory)


@pytest.fixture
def lexicon(session_factory):
    return Lexicon(session_factory, None, None, Repository(session_factory))


async def test_create_theme_dedups_by_key(repo):
    a = await repo.create_theme("Harry Potter", "speak like a wizard")
    b = await repo.create_theme("  harry  potter  ", "different prompt ignored")
    assert a.id == b.id
    assert a.theme_key == b.theme_key == "harry potter"
    # First-seen name/style_prompt win.
    assert b.name == "Harry Potter"
    assert b.style_prompt == "speak like a wizard"


async def test_create_theme_empty_key_raises(repo):
    with pytest.raises(ValueError):
        await repo.create_theme("   ", "prompt")


async def test_list_themes(repo):
    await repo.create_theme("Zorro", "z")
    await repo.create_theme("Alpha", "a")
    themes = await repo.list_themes()
    # Name-sorted.
    assert [t.name for t in themes] == ["Alpha", "Zorro"]


async def test_api_create_and_list_theme(lexicon):
    created = await lexicon.create_theme(
        "Humorous", "be funny", description="funny voice", tone="funny,playful"
    )
    assert created.key == "humorous"
    assert created.name == "Humorous"
    assert created.style_prompt == "be funny"
    assert created.description == "funny voice"
    assert created.tone == "funny,playful"

    listed = await lexicon.list_themes()
    assert [t.key for t in listed] == ["humorous"]
