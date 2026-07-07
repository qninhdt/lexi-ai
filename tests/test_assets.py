"""Tests for the reference-addressed asset cache (Phase 1: hash-verified).

Identity is ``(source_kind, source_id, kind, params)`` with ``content_hash``
VERIFIED on read: a reused/regenerated source id whose current text no longer
matches yields a MISS (regenerate), never poisoned content. Covers the hashing
verify contract, param normalization, the reshaped schema compile, reference-keyed
get/put with hash-verify, best-effort GC on the caller's session, and the
translate/tts API surface. In-memory SQLite + StaticPool; files under tmp_path.
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
from lexi_ai.models import Sense, Word
from lexi_ai.persistence.repository import Repository

# --- hashing verify contract ----------------------------------------------


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


def test_assets_table_has_reference_columns():
    from lexi_ai.models import Base

    cols = Base.metadata.tables["assets"].columns
    assert "source_kind" in cols
    assert "source_id" in cols
    assert "content_hash" in cols  # survives as a VERIFY column, not identity


# --- source-kind wiring completeness --------------------------------------


def test_every_source_kind_has_a_resolver():
    from lexi_ai.assets.repository import _SOURCE_TABLES
    from lexi_ai.constants import SOURCE_KINDS

    # A kind can never be half-wired: every SOURCE_KINDS member must resolve.
    assert set(_SOURCE_TABLES) == set(SOURCE_KINDS)


# --- repository fixtures --------------------------------------------------


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


_word_seq = 0


async def _make_sense(session_factory, definition: str = "a small domestic cat") -> int:
    """Insert a minimal word+sense; return the sense id (the source_id).

    Each call uses a unique match_key so multiple words can coexist (words.match_key
    is UNIQUE — the caller cares about sense ids, not the headword)."""
    global _word_seq
    _word_seq += 1
    async with session_scope(session_factory) as session:
        word = Word(norm=f"cat{_word_seq}", match_key=f"cat{_word_seq}", status="done")
        session.add(word)
        await session.flush()
        sense = Sense(word_id=word.id, definition=definition, tier="core")
        session.add(sense)
        await session.flush()
        return sense.id


# --- reference-keyed put/get with hash verify ------------------------------


async def test_put_text_then_get(session_factory, assets):
    sid = await _make_sense(session_factory)
    text = "a small domestic cat"
    params = normalize_asset_params("translate", lang="vi")
    assert await assets.get("sense_def", sid, "translate", params, text) is None  # miss

    put = await assets.put_text("sense_def", sid, "translate", params, text, "con mèo")
    assert put.text_value == "con mèo"
    assert put.ready

    got = await assets.get("sense_def", sid, "translate", params, text)
    assert got is not None
    assert got.text_value == "con mèo"


async def test_get_verifies_hash_stale_source_is_miss(session_factory, assets):
    # Anti-poison: a translation was cached for a sense; the sense's text then
    # changes (or its id is reused). get() verifies against the CURRENT text and
    # returns a MISS, never the stale value.
    sid = await _make_sense(session_factory, "old definition")
    params = normalize_asset_params("translate", lang="vi")
    await assets.put_text("sense_def", sid, "translate", params, "old definition", "cũ")

    # Same reference key, but the current source text is different now.
    assert await assets.get("sense_def", sid, "translate", params, "NEW definition") is None
    # Old text still verifies (unchanged) — hit.
    hit = await assets.get("sense_def", sid, "translate", params, "old definition")
    assert hit is not None and hit.text_value == "cũ"


async def test_put_text_overwrites_stale_row(session_factory, assets):
    sid = await _make_sense(session_factory)
    params = normalize_asset_params("translate", lang="vi")
    await assets.put_text("sense_def", sid, "translate", params, "old", "cũ")
    # Regenerate with new source text + value → upsert-refresh, single row.
    await assets.put_text("sense_def", sid, "translate", params, "new", "mới")
    async with session_scope(session_factory) as session:
        rows = (await session.execute(select(AssetRow))).scalars().all()
    assert len(rows) == 1
    assert rows[0].text_value == "mới"
    assert rows[0].content_hash == content_hash("new")


async def test_distinct_source_ids_no_dedup(session_factory, assets):
    # Two different sources with IDENTICAL text get SEPARATE rows (reference
    # identity, no content dedup).
    s1 = await _make_sense(session_factory, "same text")
    s2 = await _make_sense(session_factory, "same text")
    params = normalize_asset_params("translate", lang="vi")
    await assets.put_text("sense_def", s1, "translate", params, "same text", "A")
    await assets.put_text("sense_def", s2, "translate", params, "same text", "B")
    async with session_scope(session_factory) as session:
        rows = (await session.execute(select(AssetRow))).scalars().all()
    assert len(rows) == 2


async def test_put_text_rejects_nul(session_factory, assets):
    sid = await _make_sense(session_factory)
    params = normalize_asset_params("translate", lang="vi")
    with pytest.raises(ValueError):
        await assets.put_text("sense_def", sid, "translate", params, "x", "bad\x00value")


async def test_bad_source_kind_raises(assets):
    with pytest.raises(ValueError):
        await assets.get("bogus", 1, "translate", "vi", "x")


async def test_resolve_source_text(session_factory, assets):
    sid = await _make_sense(session_factory, "resolvable text")
    assert await assets.resolve_source_text("sense_def", sid) == "resolvable text"
    assert await assets.resolve_source_text("sense_def", 999999) is None


# --- file put/shard -------------------------------------------------------


async def test_put_file_shards_and_stores_relative(session_factory, assets, tmp_path):
    sid = await _make_sense(session_factory)
    text = "a small domestic cat"
    params = normalize_asset_params("tts", voice="alloy", fmt="mp3")
    put = await assets.put_file("sense_def", sid, "tts", params, text, b"\x01\x02\x03", ext="mp3")
    h = content_hash(text)
    assert put.file_path == f"{h[:2]}/{h}.alloy-mp3.mp3"
    assert (tmp_path / put.file_path).read_bytes() == b"\x01\x02\x03"

    got = await assets.get("sense_def", sid, "tts", params, text)
    assert got is not None
    assert got.file_path == put.file_path
    assert got.ready


async def test_get_missing_file_is_miss(session_factory, assets, tmp_path):
    sid = await _make_sense(session_factory)
    text = "a small domestic cat"
    params = normalize_asset_params("tts", voice="alloy", fmt="mp3")
    put = await assets.put_file("sense_def", sid, "tts", params, text, b"\x01", ext="mp3")
    (tmp_path / put.file_path).unlink()
    assert await assets.get("sense_def", sid, "tts", params, text) is None


async def test_put_file_path_traversal_is_contained(session_factory, assets, tmp_path):
    # A voice/fmt with path separators must NOT escape the shard dir.
    sid = await _make_sense(session_factory)
    text = "a small domestic cat"
    params = normalize_asset_params("tts", voice="../evil", fmt="mp3")
    put = await assets.put_file("sense_def", sid, "tts", params, text, b"\x01", ext="mp3")
    resolved = (tmp_path / put.file_path).resolve()
    assert str(resolved).startswith(str(tmp_path.resolve()))


# --- best-effort GC on the caller's session --------------------------------


async def test_delete_by_source_removes_row_and_file(session_factory, assets, tmp_path):
    sid = await _make_sense(session_factory)
    text = "a small domestic cat"
    p_t = normalize_asset_params("translate", lang="vi")
    p_a = normalize_asset_params("tts", voice="alloy", fmt="mp3")
    await assets.put_text("sense_def", sid, "translate", p_t, text, "con mèo")
    file_asset = await assets.put_file("sense_def", sid, "tts", p_a, text, b"\x01", ext="mp3")
    file_path = tmp_path / file_asset.file_path
    assert file_path.exists()

    async with session_scope(session_factory) as session:
        n = await assets.delete_by_source(session, "sense_def", sid)
    assert n == 2
    async with session_scope(session_factory) as session:
        rows = (await session.execute(select(AssetRow))).scalars().all()
    assert rows == []
    assert not file_path.exists()  # file unlinked


async def test_gc_rolls_back_with_caller_transaction(session_factory, assets):
    # GC runs on the caller's session, so if the transaction fails AFTER gc, the
    # gc deletes roll back too (no independent commit).
    sid = await _make_sense(session_factory)
    text = "a small domestic cat"
    p_t = normalize_asset_params("translate", lang="vi")
    await assets.put_text("sense_def", sid, "translate", p_t, text, "con mèo")

    with pytest.raises(RuntimeError):
        async with session_scope(session_factory) as session:
            await assets.delete_by_source(session, "sense_def", sid)
            raise RuntimeError("boom after gc")
    # Row survives — gc was part of the rolled-back transaction.
    async with session_scope(session_factory) as session:
        rows = (await session.execute(select(AssetRow))).scalars().all()
    assert len(rows) == 1


# --- API surface: translate ------------------------------------------------


class FakeTranslator:
    """Returns ``{lang}:{text}``; counts calls to prove cache hits skip the LLM."""

    def __init__(self):
        self.calls = 0

    async def translate(self, text: str, lang: str) -> str:
        self.calls += 1
        return f"{lang}:{text}"


def _lexicon(session_factory, tmp_path, *, translator=None, tts=None):
    from lexi_ai.api import Lexicon

    assets = AssetRepository(session_factory, str(tmp_path))
    lex = Lexicon(
        session_factory,
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        Repository(session_factory, assets=assets),
        assets=assets,
    )
    if translator is not None:
        lex._translator_impl = translator
    if tts is not None:
        lex._tts_impl = tts
    return lex


async def test_translate_sense_caches(session_factory, tmp_path):
    sid = await _make_sense(session_factory, "a small domestic cat")
    tr = FakeTranslator()
    lex = _lexicon(session_factory, tmp_path, translator=tr)
    first = await lex.translate_sense(sid, "vi")
    second = await lex.translate_sense(sid, "vi")
    assert first == second == "vi:a small domestic cat"
    assert tr.calls == 1  # second call is a cache hit


async def test_translate_field_delegates_to_sense_def(session_factory, tmp_path):
    sid = await _make_sense(session_factory, "def text")
    tr = FakeTranslator()
    lex = _lexicon(session_factory, tmp_path, translator=tr)
    assert await lex.translate_field("sense_def", sid, "vi") == "vi:def text"


async def test_translate_bad_ref_raises(session_factory, tmp_path):
    tr = FakeTranslator()
    lex = _lexicon(session_factory, tmp_path, translator=tr)
    with pytest.raises(ValueError):
        await lex.translate_field("sense_def", 999999, "vi")
    assert tr.calls == 0


async def test_translate_no_llm_raises(session_factory, tmp_path, monkeypatch):
    sid = await _make_sense(session_factory)
    from lexi_ai.api import Lexicon

    assets = AssetRepository(session_factory, str(tmp_path))
    lex = Lexicon(
        session_factory,
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        Repository(session_factory, assets=assets),
        assets=assets,
    )
    monkeypatch.setenv("LEXI_LLM_API_KEY", "")
    with pytest.raises(ValueError):
        await lex.translate_sense(sid, "vi")


# --- API surface: tts (stub + fake real provider) --------------------------


class FakeRealTTS:
    """A fake 'real' provider returning bytes — proves the put_file path."""

    def __init__(self):
        self.calls = 0

    async def synthesize(self, text: str, voice: str, fmt: str) -> bytes:
        self.calls += 1
        return f"AUDIO({voice}/{fmt}):{text}".encode()


async def test_tts_sense_real_provider_round_trip(session_factory, tmp_path):
    sid = await _make_sense(session_factory, "a small domestic cat")
    provider = FakeRealTTS()
    lex = _lexicon(session_factory, tmp_path, tts=provider)
    first = await lex.tts_sense(sid, voice="alloy", fmt="mp3")
    assert first.file_path is not None
    assert (tmp_path / first.file_path).read_bytes() == b"AUDIO(alloy/mp3):a small domestic cat"
    second = await lex.tts_sense(sid, voice="alloy", fmt="mp3")
    assert second.file_path == first.file_path
    assert provider.calls == 1  # cache hit


async def test_tts_stub_miss_raises_no_row(session_factory, tmp_path):
    sid = await _make_sense(session_factory)
    lex = _lexicon(session_factory, tmp_path)  # default StubTTSProvider
    with pytest.raises(NotImplementedError):
        await lex.tts_sense(sid, voice="alloy", fmt="mp3")
    async with session_scope(session_factory) as session:
        rows = (await session.execute(select(AssetRow))).scalars().all()
    assert rows == []
    assert not any(tmp_path.rglob("*.mp3"))


async def test_tts_empty_source_short_circuits(session_factory, tmp_path):
    sid = await _make_sense(session_factory, "   ")
    provider = FakeRealTTS()
    lex = _lexicon(session_factory, tmp_path, tts=provider)
    asset = await lex.tts_sense(sid, voice="alloy", fmt="mp3")
    assert not asset.ready
    assert provider.calls == 0


# --- OpenAI-compatible TTS provider (fake client, no network) --------------


class _FakeBinaryResponse:
    """Stands in for the SDK's HttpxBinaryResponseContent — ``aread`` → bytes."""

    def __init__(self, data: bytes):
        self._data = data

    async def aread(self) -> bytes:
        return self._data


class _FakeSpeech:
    def __init__(self, sink: dict):
        self._sink = sink

    async def create(self, **kwargs):
        self._sink.update(kwargs)
        return _FakeBinaryResponse(b"FAKEAUDIO")


class _FakeAudio:
    def __init__(self, sink: dict):
        self.speech = _FakeSpeech(sink)


class _FakeOpenAIClient:
    """Minimal AsyncOpenAI stand-in exposing ``audio.speech.create``."""

    def __init__(self, sink: dict):
        self.audio = _FakeAudio(sink)


async def test_tts_provider_builds_request_from_settings():
    from lexi_ai.assets.tts import OpenAICompatibleTTSProvider

    sink: dict = {}
    provider = OpenAICompatibleTTSProvider(
        base_url="https://tts.example/v1",
        api_key="sk-test",
        model="tts-1",
        client=_FakeOpenAIClient(sink),
    )
    data = await provider.synthesize("hello", "alloy", "mp3")
    assert data == b"FAKEAUDIO"
    assert sink["model"] == "tts-1"
    assert sink["input"] == "hello"
    assert sink["voice"] == "alloy"
    assert sink["response_format"] == "mp3"


def test_tts_base_url_non_https_with_key_raises():
    from lexi_ai.assets.tts import OpenAICompatibleTTSProvider

    with pytest.raises(ValueError, match="non-https"):
        OpenAICompatibleTTSProvider(
            base_url="http://tts.example/v1", api_key="sk-test", model="tts-1"
        )


def test_tts_base_url_empty_with_key_raises():
    from lexi_ai.assets.tts import OpenAICompatibleTTSProvider

    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatibleTTSProvider(base_url="", api_key="sk-test", model="tts-1")


def test_tts_base_url_loopback_http_allowed():
    from lexi_ai.assets.tts import OpenAICompatibleTTSProvider

    # Should NOT raise — loopback http never ships the key over the wire.
    OpenAICompatibleTTSProvider(
        base_url="http://127.0.0.1:8080/v1",
        api_key="sk-test",
        model="tts-1",
        client=_FakeOpenAIClient({}),
    )


def test_tts_no_key_skips_base_url_check():
    from lexi_ai.assets.tts import OpenAICompatibleTTSProvider

    # No key set → nothing to leak → a plain http base_url is fine to construct.
    OpenAICompatibleTTSProvider(
        base_url="http://tts.example/v1",
        api_key="",
        model="tts-1",
        client=_FakeOpenAIClient({}),
    )


async def test_tts_provider_selection_real_when_configured(
    session_factory, tmp_path, monkeypatch
):
    from lexi_ai.assets.tts import OpenAICompatibleTTSProvider

    monkeypatch.setenv("LEXI_TTS_BASE_URL", "https://tts.example/v1")
    monkeypatch.setenv("LEXI_TTS_API_KEY", "sk-test")
    monkeypatch.setenv("LEXI_TTS_MODEL", "tts-1")
    lex = _lexicon(session_factory, tmp_path)  # no injected provider
    assert isinstance(lex._tts_provider(), OpenAICompatibleTTSProvider)


async def test_tts_provider_selection_stub_when_unconfigured(
    session_factory, tmp_path, monkeypatch
):
    from lexi_ai.assets.tts import StubTTSProvider

    monkeypatch.setenv("LEXI_TTS_BASE_URL", "")
    monkeypatch.setenv("LEXI_TTS_API_KEY", "")
    lex = _lexicon(session_factory, tmp_path)
    assert isinstance(lex._tts_provider(), StubTTSProvider)


async def test_tts_provider_selection_real_when_only_base_url(
    session_factory, tmp_path, monkeypatch
):
    """A base_url set with an empty key still selects the real (keyless) provider —
    for local keyless compat servers. The guard skips (nothing to leak)."""
    from lexi_ai.assets.tts import OpenAICompatibleTTSProvider

    monkeypatch.setenv("LEXI_TTS_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("LEXI_TTS_API_KEY", "")
    lex = _lexicon(session_factory, tmp_path)
    assert isinstance(lex._tts_provider(), OpenAICompatibleTTSProvider)


async def test_tts_empty_audio_raises_no_cache(session_factory, tmp_path):
    """A 200-with-empty-body response must not be cached as valid audio."""

    class _EmptyResponse:
        async def aread(self) -> bytes:
            return b""

    class _EmptySpeech:
        async def create(self, **kwargs):
            return _EmptyResponse()

    class _EmptyClient:
        def __init__(self):
            self.audio = type("_A", (), {"speech": _EmptySpeech()})()

    from lexi_ai.assets.tts import OpenAICompatibleTTSProvider

    provider = OpenAICompatibleTTSProvider(
        base_url="https://tts.example/v1",
        api_key="sk-test",
        model="tts-1",
        client=_EmptyClient(),
    )
    with pytest.raises(ValueError, match="empty"):
        await provider.synthesize("hello", "alloy", "mp3")


def test_tts_base_url_uppercase_https_with_key_allowed():
    """Scheme casing must not defeat the https check (urlparse lowercases it)."""
    from lexi_ai.assets.tts import OpenAICompatibleTTSProvider

    OpenAICompatibleTTSProvider(
        base_url="HTTPS://tts.example/v1",
        api_key="sk-test",
        model="tts-1",
        client=_FakeOpenAIClient({}),
    )


def test_tts_base_url_missing_scheme_with_key_raises():
    from lexi_ai.assets.tts import OpenAICompatibleTTSProvider

    with pytest.raises(ValueError):
        OpenAICompatibleTTSProvider(
            base_url="tts.example/v1", api_key="sk-test", model="tts-1"
        )


def test_tts_base_url_ipv6_loopback_http_allowed():
    from lexi_ai.assets.tts import OpenAICompatibleTTSProvider

    OpenAICompatibleTTSProvider(
        base_url="http://[::1]:8080/v1",
        api_key="sk-test",
        model="tts-1",
        client=_FakeOpenAIClient({}),
    )


# --- API surface: tts_many (batch mirror of translate_many) ----------------


async def test_tts_many_order_aligned(session_factory, tmp_path):
    s1 = await _make_sense(session_factory, "the first definition")
    s2 = await _make_sense(session_factory, "the second definition")
    provider = FakeRealTTS()
    lex = _lexicon(session_factory, tmp_path, tts=provider)
    refs = [("sense_def", s1), ("sense_def", s2)]
    results = await lex.tts_many(refs, voice="alloy", fmt="mp3")
    # One order-aligned BatchResult per ref, all ok, key echoes the input.
    assert [r.key for r in results] == refs
    assert all(r.ok for r in results)
    assert all(r.value.file_path is not None for r in results)
    assert provider.calls == 2


async def test_tts_many_cache_first_per_ref(session_factory, tmp_path):
    # A ref already synthesized (cached) is not re-synthesized on a later batch —
    # the same cache-first guarantee tts_field carries, per item.
    sid = await _make_sense(session_factory, "a cached definition")
    provider = FakeRealTTS()
    lex = _lexicon(session_factory, tmp_path, tts=provider)
    await lex.tts_field("sense_def", sid, voice="alloy", fmt="mp3")
    assert provider.calls == 1
    results = await lex.tts_many([("sense_def", sid)], voice="alloy", fmt="mp3")
    assert results[0].ok and results[0].value.file_path is not None
    assert provider.calls == 1  # served from cache, no new synth


async def test_tts_many_empty_is_empty(session_factory, tmp_path):
    lex = _lexicon(session_factory, tmp_path, tts=FakeRealTTS())
    assert await lex.tts_many([]) == []


async def test_tts_many_one_failure_does_not_abort(session_factory, tmp_path):
    sid = await _make_sense(session_factory, "a real definition")
    provider = FakeRealTTS()
    lex = _lexicon(session_factory, tmp_path, tts=provider)
    # A bad ref (no source row) fails that item only; the good one still resolves.
    results = await lex.tts_many([("sense_def", 999999), ("sense_def", sid)])
    assert not results[0].ok and results[0].error
    assert results[1].ok and results[1].value.file_path is not None


async def test_tts_many_stub_reports_per_item_error(session_factory, tmp_path):
    # Unconfigured TTS (stub) raises per item; the batch reports it, never aborts.
    sid = await _make_sense(session_factory, "a definition")
    lex = _lexicon(session_factory, tmp_path)  # default StubTTSProvider
    results = await lex.tts_many([("sense_def", sid)])
    assert len(results) == 1
    assert not results[0].ok and results[0].error


# --- GC via the persistence delete/regenerate paths ------------------------


async def test_delete_word_gcs_assets(session_factory, tmp_path):
    sid = await _make_sense(session_factory, "a small domestic cat")
    assets = AssetRepository(session_factory, str(tmp_path))
    repo = Repository(session_factory, assets=assets)
    p = normalize_asset_params("translate", lang="vi")
    await assets.put_text("sense_def", sid, "translate", p, "a small domestic cat", "con mèo")

    # Find the owning word id and delete it.
    async with session_scope(session_factory) as session:
        wid = (await session.execute(select(Sense.word_id).where(Sense.id == sid))).scalar_one()
    assert await repo.delete_word(wid) is True

    async with session_scope(session_factory) as session:
        rows = (await session.execute(select(AssetRow))).scalars().all()
    assert rows == []  # asset GC'd with the word
