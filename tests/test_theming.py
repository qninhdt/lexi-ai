"""Tests for themed generation + read overlay (Phases 2-3).

Covers the ThemedResult schema, the prompt formatter (neutral facts present,
neutral examples absent, senses numbered), repository persist/read (overwrite,
count-mismatch guard, overlay map), and the api ``generate_theme`` + themed
``get`` overlay with a fake generator (no network). In-memory SQLite + StaticPool.
"""

import pytest
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from lexi_ai.api import Lexicon
from lexi_ai.db import create_session_factory, init_models, session_scope
from lexi_ai.models import Example, Sense, ThemedExample, ThemedSense, Word
from lexi_ai.persistence.repository import Repository
from lexi_ai.read_models import Entry
from lexi_ai.theming.prompts import format_themed
from lexi_ai.theming.schemas import ThemedResult, GeneratedTheme
from lexi_ai.theming.schemas import ThemedSense as ThemedSenseSchema


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


async def _make_done_word(session_factory, n_senses=2):
    """Insert one done word with ``n_senses`` neutral senses (each with an example)."""
    async with session_scope(session_factory) as session:
        senses = [
            Sense(
                definition=f"neutral def {i}",
                tier="core",
                sense_order=i,
                pos="noun",
                guideword=f"GW{i}",
                examples=[Example(text=f"neutral example {i}", example_order=0)],
            )
            for i in range(n_senses)
        ]
        word = Word(norm="dragon", match_key="dragon", status="done", senses=senses)
        session.add(word)
        await session.flush()
        return word.id, [s.id for s in sorted(word.senses, key=lambda s: s.sense_order)]


# --- schema ---------------------------------------------------------------


def test_themed_result_rejects_empty_senses():
    with pytest.raises(ValueError):
        ThemedResult(senses=[])


def test_themed_sense_defaults_examples_empty():
    s = ThemedSenseSchema(definition="d")
    assert s.examples == []


# --- prompt formatter -----------------------------------------------------


def test_format_themed_has_facts_not_neutral_examples():
    neutral = [
        ("a scaly beast", "noun", "CREATURE", "core"),
        ("to hoard", None, None, "common"),
    ]
    prompt = format_themed("speak like a bard", neutral)
    # Facts present.
    assert "a scaly beast" in prompt
    assert "CREATURE" in prompt
    assert "core" in prompt
    assert "noun" in prompt
    assert "speak like a bard" in prompt
    # Senses numbered 1..N.
    assert "Sense 1:" in prompt
    assert "Sense 2:" in prompt
    # Optional fields omitted cleanly for sense 2 (no guideword line duplicated).
    assert prompt.count("guideword:") == 1


# --- persist_themed -------------------------------------------------------


async def test_persist_themed_happy_and_overwrite(session_factory, repo):
    word_id, sense_ids = await _make_done_word(session_factory)
    theme = await repo.create_theme("Bard", "speak like a bard")

    result = ThemedResult(
        senses=[
            ThemedSenseSchema(definition="a mighty wyrm", examples=["Lo, a wyrm!", "Behold!"]),
            ThemedSenseSchema(definition="to amass treasure", examples=["Hoard ye gold."]),
        ]
    )
    await repo.persist_themed(theme.id, result, sense_ids)

    overlay = await repo.themed_for_word(word_id, theme.id)
    assert overlay[sense_ids[0]][0] == "a mighty wyrm"
    assert overlay[sense_ids[0]][1] == ["Lo, a wyrm!", "Behold!"]
    assert overlay[sense_ids[1]][0] == "to amass treasure"

    # Re-run overwrites in place (no dup under UNIQUE(sense_id, theme_id)).
    result2 = ThemedResult(
        senses=[
            ThemedSenseSchema(definition="a re-styled wyrm", examples=["New."]),
            ThemedSenseSchema(definition="to re-amass", examples=[]),
        ]
    )
    await repo.persist_themed(theme.id, result2, sense_ids)
    async with session_scope(session_factory) as session:
        ts_count = len((await session.execute(select(ThemedSense))).all())
        ex_count = len((await session.execute(select(ThemedExample))).all())
    assert ts_count == 2  # still one row per sense
    assert ex_count == 1  # only the single new example survived
    overlay2 = await repo.themed_for_word(word_id, theme.id)
    assert overlay2[sense_ids[0]][0] == "a re-styled wyrm"
    assert overlay2[sense_ids[1]][1] == []


async def test_persist_themed_count_mismatch_raises(session_factory, repo):
    _word_id, sense_ids = await _make_done_word(session_factory)
    theme = await repo.create_theme("Bard", "voice")
    result = ThemedResult(senses=[ThemedSenseSchema(definition="only one")])
    with pytest.raises(ValueError):
        await repo.persist_themed(theme.id, result, sense_ids)


# --- generate_theme + overlay read (api) ----------------------------------


class FakeThemedGenerator:
    """Returns a canned ThemedResult; records the style prompt + facts it saw."""

    def __init__(self, result: ThemedResult):
        self._result = result
        self.calls = 0
        self.last_style = None
        self.last_facts = None

    async def generate(self, style_prompt, neutral_senses):
        self.calls += 1
        self.last_style = style_prompt
        self.last_facts = list(neutral_senses)
        return self._result


class FakeThemeMetadataGenerator:
    """Returns a canned GeneratedTheme."""

    def __init__(self, result: GeneratedTheme):
        self._result = result
        self.calls = 0

    async def generate(self, key, prompt):
        self.calls += 1
        return self._result


def _lexicon(session_factory, themed_gen=None, theme_meta_gen=None):
    lex = Lexicon(session_factory, None, None, Repository(session_factory))  # type: ignore[arg-type]
    if themed_gen is not None:
        lex._themed_gen = themed_gen
    if theme_meta_gen is not None:
        lex._theme_meta_gen = theme_meta_gen
    return lex


async def test_generate_theme_metadata(session_factory):
    from lexi_ai.theming.schemas import GeneratedTheme
    gen = FakeThemeMetadataGenerator(
        GeneratedTheme(
            name="The Salty Pirate Captain",
            description="Salty sea-themed dictionary entries.",
            style_prompt="Salty instructions.",
            emoji="🏴‍☠️",
            tone=["salty", "adventurous"]
        )
    )
    lex = _lexicon(session_factory, theme_meta_gen=gen)
    theme = await lex.generate_theme("pirate", "speak like a pirate")
    assert theme.name == "The Salty Pirate Captain"
    assert theme.key == "pirate"
    assert theme.description == "Salty sea-themed dictionary entries."
    assert theme.style_prompt == "Salty instructions."
    assert theme.emoji == "🏴‍☠️"
    assert theme.tone == "salty,adventurous"
    assert gen.calls == 1


async def test_generate_theme_end_to_end(session_factory):
    from lexi_ai.read_models import SearchResult
    word_id, sense_ids = await _make_done_word(session_factory)
    gen = FakeThemedGenerator(
        ThemedResult(
            senses=[
                ThemedSenseSchema(definition="themed 0", examples=["ex a"]),
                ThemedSenseSchema(definition="themed 1", examples=["ex b"]),
            ]
        )
    )
    lex = _lexicon(session_factory, gen)
    await lex.create_theme("Bard", "speak like a bard")

    source = SearchResult(display="dragon", entry_type="word", lexi_word_id=word_id)
    entry = await lex.generate(source, theme="bard")
    assert isinstance(entry, Entry)
    assert gen.calls == 1
    # The generator saw the neutral FACTS, never the neutral examples.
    assert gen.last_style == "speak like a bard"
    assert gen.last_facts[0][0] == "neutral def 0"
    # Themed overlay applied on read.
    assert entry.senses[0].definition == "themed 0"
    assert entry.senses[0].examples == ["ex a"]
    assert entry.senses[1].definition == "themed 1"


async def test_generate_theme_requires_done_word(session_factory):
    from lexi_ai.read_models import SearchResult
    async with session_scope(session_factory) as session:
        word = Word(norm="pending", match_key="pending", status="pending")
        session.add(word)
        await session.flush()
        wid = word.id
    gen = FakeThemedGenerator(ThemedResult(senses=[ThemedSenseSchema(definition="x")]))
    lex = _lexicon(session_factory, gen)
    await lex.create_theme("Bard", "voice")
    
    source = SearchResult(display="pending", entry_type="word", lexi_word_id=wid)
    with pytest.raises(ValueError, match="is not done"):
        await lex.generate(source, theme="bard")


async def test_generate_theme_unknown_theme_raises(session_factory):
    from lexi_ai.read_models import SearchResult
    word_id, _ = await _make_done_word(session_factory)
    gen = FakeThemedGenerator(ThemedResult(senses=[ThemedSenseSchema(definition="x")]))
    lex = _lexicon(session_factory, gen)
    
    source = SearchResult(display="dragon", entry_type="word", lexi_word_id=word_id)
    with pytest.raises(ValueError, match="unknown theme"):
        await lex.generate(source, theme="nonexistent")


# --- read overlay (Phase 3) -----------------------------------------------


async def test_get_neutral_unchanged(session_factory):
    word_id, _ = await _make_done_word(session_factory)
    lex = _lexicon(session_factory)
    entry = await lex.get(word_id)
    assert entry.senses[0].definition == "neutral def 0"
    assert entry.senses[0].examples == ["neutral example 0"]


async def test_get_theme_none_is_neutral(session_factory):
    word_id, _ = await _make_done_word(session_factory)
    lex = _lexicon(session_factory)
    entry = await lex.get(word_id, theme_key=None)
    assert entry.senses[0].definition == "neutral def 0"


async def test_get_unknown_theme_raises(session_factory):
    word_id, _ = await _make_done_word(session_factory)
    lex = _lexicon(session_factory)
    with pytest.raises(ValueError):
        await lex.get(word_id, theme_key="ghost")


async def test_get_per_sense_fallback(session_factory, repo):
    word_id, sense_ids = await _make_done_word(session_factory)
    theme = await repo.create_theme("Bard", "voice")
    # Theme only sense[0]: insert one themed row directly.
    async with session_scope(session_factory) as session:
        session.add(
            ThemedSense(sense_id=sense_ids[0], theme_id=theme.id, definition="themed only 0")
        )
        await session.flush()
    lex = _lexicon(session_factory)
    entry = await lex.get(word_id, theme_key="bard")
    assert entry.senses[0].definition == "themed only 0"  # themed
    assert entry.senses[1].definition == "neutral def 1"  # fallback
