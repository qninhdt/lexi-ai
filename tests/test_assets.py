"""Tests for the content-addressed asset cache (Phase 4: foundation).

Covers the hashing contract (normalization-stable, distinct text distinct hash),
param normalization (stable read==write, unknown kind raises), the Asset schema
compile, and the repository resolve-or-create + file-shard paths. In-memory
SQLite + StaticPool; files under pytest's ``tmp_path``.
"""

import pytest
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from lexi_ai.assets.repository import (
    AssetRepository,
    content_hash,
    normalize_asset_params,
)
from lexi_ai.db import create_session_factory, init_models, session_scope
from lexi_ai.models import Asset as AssetRow
from lexi_ai.persistence.repository import Repository

# --- hashing contract -----------------------------------------------------


def test_content_hash_normalization_stable():
    base = content_hash("hello world")
    assert content_hash("  hello   world  ") == base
    assert content_hash("hello\tworld") == base
    assert content_hash("hello\x00world") == base  # control stripped


def test_content_hash_distinct_text():
    assert content_hash("cat") != content_hash("dog")


def test_content_hash_is_hex_sha256():
    h = content_hash("x")
    assert len(h) == 64
    int(h, 16)  # hex-decodable


# --- param normalization --------------------------------------------------


def test_translate_params_stable():
    assert normalize_asset_params("translate", lang="vi") == "vi"
    assert normalize_asset_params("translate", lang=" VI ") == "vi"


def test_translate_params_invalid_raises():
    with pytest.raises(ValueError, match="invalid/unsupported language code"):
        normalize_asset_params("translate", lang="invalid_code")


def test_tts_params_stable():
    assert normalize_asset_params("tts", voice="Alloy", fmt="MP3") == "alloy|mp3"
    assert normalize_asset_params("tts", voice=None, fmt=None) == "|"


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        normalize_asset_params("bogus", lang="vi")


# --- schema compiles on both dialects -------------------------------------


def test_asset_table_compiles_on_both_dialects():
    from sqlalchemy.dialects import postgresql, sqlite
    from sqlalchemy.schema import CreateTable

    from lexi_ai.models import Base

    assert "assets" in Base.metadata.tables
    for dialect in (postgresql.dialect(), sqlite.dialect()):
        for table in Base.metadata.sorted_tables:
            ddl = str(CreateTable(table).compile(dialect=dialect))
            assert "CREATE TABLE" in ddl


# --- repository -----------------------------------------------------------


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
def assets(session_factory, tmp_path):
    return AssetRepository(session_factory, str(tmp_path))


async def test_put_text_then_get(assets):
    h = content_hash("cat")
    params = normalize_asset_params("translate", lang="vi")
    assert await assets.get(h, "translate", params) is None  # miss

    put = await assets.put_text(h, "translate", params, "con mèo")
    assert put.text_value == "con mèo"
    assert put.ready

    got = await assets.get(h, "translate", params)
    assert got is not None
    assert got.text_value == "con mèo"


async def test_put_text_dedups(session_factory, assets):
    h = content_hash("cat")
    params = normalize_asset_params("translate", lang="vi")
    await assets.put_text(h, "translate", params, "con mèo")
    await assets.put_text(h, "translate", params, "ignored second value")
    async with session_scope(session_factory) as session:
        rows = (await session.execute(select(AssetRow))).all()
    assert len(rows) == 1  # one row under UNIQUE identity


async def test_put_text_rejects_nul(assets):
    h = content_hash("cat")
    params = normalize_asset_params("translate", lang="vi")
    with pytest.raises(ValueError):
        await assets.put_text(h, "translate", params, "bad\x00value")


async def test_put_file_shards_and_stores_relative(assets, tmp_path):
    h = content_hash("hello")
    params = normalize_asset_params("tts", voice="alloy", fmt="mp3")
    put = await assets.put_file(h, "tts", params, b"\x01\x02\x03", ext="mp3")
    # Relative path, sharded by hash prefix, with params folded in (voice|fmt).
    assert put.file_path == f"{h[:2]}/{h}.alloy-mp3.mp3"
    assert (tmp_path / put.file_path).read_bytes() == b"\x01\x02\x03"

    got = await assets.get(h, "tts", params)
    assert got is not None
    assert got.file_path == put.file_path
    assert got.ready


async def test_put_file_idempotent(session_factory, assets):
    h = content_hash("hello")
    params = normalize_asset_params("tts", voice="alloy", fmt="mp3")
    await assets.put_file(h, "tts", params, b"\x01", ext="mp3")
    await assets.put_file(h, "tts", params, b"\x02", ext="mp3")
    async with session_scope(session_factory) as session:
        rows = (await session.execute(select(AssetRow))).all()
    assert len(rows) == 1


async def test_get_missing_file_is_miss(session_factory, assets, tmp_path):
    h = content_hash("hello")
    params = normalize_asset_params("tts", voice="alloy", fmt="mp3")
    put = await assets.put_file(h, "tts", params, b"\x01", ext="mp3")
    # Delete the file behind the row → get treats it as a miss (defensive).
    (tmp_path / put.file_path).unlink()
    assert await assets.get(h, "tts", params) is None


async def test_put_file_same_text_different_params_no_collision(assets, tmp_path):
    # Same text + fmt, DIFFERENT voice → distinct files (params folded into path),
    # so neither voice's bytes overwrite the other's.
    h = content_hash("hello")
    p_alloy = normalize_asset_params("tts", voice="alloy", fmt="mp3")
    p_nova = normalize_asset_params("tts", voice="nova", fmt="mp3")
    a = await assets.put_file(h, "tts", p_alloy, b"ALLOY", ext="mp3")
    n = await assets.put_file(h, "tts", p_nova, b"NOVA", ext="mp3")
    assert a.file_path != n.file_path
    assert (tmp_path / a.file_path).read_bytes() == b"ALLOY"
    assert (tmp_path / n.file_path).read_bytes() == b"NOVA"


# --- translation (Phase 5) ------------------------------------------------


class FakeTranslator:
    """Returns ``{lang}:{text}``; counts calls to prove cache hits skip the LLM."""

    def __init__(self):
        self.calls = 0

    async def translate(self, text: str, lang: str) -> str:
        self.calls += 1
        return f"{lang}:{text}"


def _lexicon_with_translator(session_factory, tmp_path, translator):
    from lexi_ai.api import Lexicon

    lex = Lexicon(
        session_factory,
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        Repository(session_factory),
        assets=AssetRepository(session_factory, str(tmp_path)),
    )
    lex._translator_impl = translator
    return lex


async def test_translate_caches(session_factory, tmp_path):
    from lexi_ai.persistence.repository import Repository as _R  # noqa: F401

    tr = FakeTranslator()
    lex = _lexicon_with_translator(session_factory, tmp_path, tr)
    first = await lex.translate("cat", "vi")
    second = await lex.translate("cat", "vi")
    assert first == second == "vi:cat"
    assert tr.calls == 1  # second call is a cache hit


async def test_translate_distinct_source_distinct_asset(session_factory, tmp_path):
    tr = FakeTranslator()
    lex = _lexicon_with_translator(session_factory, tmp_path, tr)
    await lex.translate("cat", "vi")
    await lex.translate("dog", "vi")  # different source → new call
    await lex.translate("cat", "vi")  # shared with first
    assert tr.calls == 2


async def test_translate_empty_short_circuits(session_factory, tmp_path):
    tr = FakeTranslator()
    lex = _lexicon_with_translator(session_factory, tmp_path, tr)
    assert await lex.translate("   ", "vi") == "   "
    assert tr.calls == 0


async def test_translate_no_llm_raises(session_factory, tmp_path, monkeypatch):
    from lexi_ai.api import Lexicon

    lex = Lexicon(
        session_factory,
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        Repository(session_factory),
        assets=AssetRepository(session_factory, str(tmp_path)),
    )
    # No injected translator + no api key → raise.
    monkeypatch.setenv("LEXI_LLM_API_KEY", "")
    with pytest.raises(ValueError):
        await lex.translate("cat", "vi")


# --- TTS (Phase 6: stub + interface) --------------------------------------


class FakeRealTTS:
    """A fake 'real' provider returning bytes — proves the future put_file path."""

    def __init__(self):
        self.calls = 0

    async def synthesize(self, text: str, voice: str, fmt: str) -> bytes:
        self.calls += 1
        return f"AUDIO({voice}/{fmt}):{text}".encode()


def _lexicon_for_tts(session_factory, tmp_path, provider=None):
    from lexi_ai.api import Lexicon

    lex = Lexicon(
        session_factory,
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        Repository(session_factory),
        assets=AssetRepository(session_factory, str(tmp_path)),
    )
    if provider is not None:
        lex._tts_impl = provider
    return lex


async def test_tts_cache_hit_skips_provider(session_factory, tmp_path):
    provider = FakeRealTTS()
    lex = _lexicon_for_tts(session_factory, tmp_path, provider)
    params = normalize_asset_params("tts", voice="alloy", fmt="mp3")
    h = content_hash("hello")
    # Pre-seed a file asset directly, then tts must NOT call the provider.
    await lex._require_assets().put_file(h, "tts", params, b"seed", ext="mp3")
    got = await lex.tts("hello", voice="alloy", fmt="mp3")
    assert got.file_path == f"{h[:2]}/{h}.alloy-mp3.mp3"
    assert provider.calls == 0


async def test_tts_stub_miss_raises_no_row(session_factory, tmp_path):
    lex = _lexicon_for_tts(session_factory, tmp_path)  # default StubTTSProvider
    with pytest.raises(NotImplementedError):
        await lex.tts("brand new text", voice="alloy", fmt="mp3")
    # No row, no file written on a stubbed miss.
    async with session_scope(session_factory) as session:
        rows = (await session.execute(select(AssetRow))).all()
    assert rows == []
    assert not any(tmp_path.rglob("*.mp3"))


async def test_tts_empty_short_circuits(session_factory, tmp_path):
    provider = FakeRealTTS()
    lex = _lexicon_for_tts(session_factory, tmp_path, provider)
    asset = await lex.tts("   ", voice="alloy", fmt="mp3")
    assert not asset.ready
    assert provider.calls == 0


async def test_tts_real_provider_round_trip(session_factory, tmp_path):
    provider = FakeRealTTS()
    lex = _lexicon_for_tts(session_factory, tmp_path, provider)
    first = await lex.tts("hello", voice="alloy", fmt="mp3")
    assert first.file_path is not None
    assert (tmp_path / first.file_path).read_bytes() == b"AUDIO(alloy/mp3):hello"
    # Second call is a cache hit — provider not called again.
    second = await lex.tts("hello", voice="alloy", fmt="mp3")
    assert second.file_path == first.file_path
    assert provider.calls == 1
