"""Derived assets: translations and synthesized speech, cache-first.

Both surfaces follow one rule: resolve the CURRENT source text, then read the cache
verified against it. The cache key is the reference tuple, not a content hash, so a
regenerated source id would otherwise serve the previous text; verifying on read is what
makes a reused id a miss instead of poisoned content. A missing translator raises, since a
caller asking for a translation cannot be served silently; a missing speech provider is the
stub, which raises rather than caching fake audio.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

from lexi_ai.dictionary.batch import gather_batch
from lexi_ai.models import Asset, BatchResult
from lexi_ai.text import content_hash, normalize_asset_params

if TYPE_CHECKING:
    from lexi_ai.db.queries.asset import AssetRepository


class AssetService:
    def __init__(
        self,
        assets: AssetRepository,
        translator_factory: Callable[[], object | None],
        tts_factory: Callable[[], object],
        voice: str,
        fmt: str,
        gather: Callable[..., Awaitable[list[BatchResult]]] = gather_batch,
    ) -> None:
        self._assets = assets
        # Factories rather than instances: a provider is built from settings on first
        # use, and a test injects a fake via its own factory.
        self._translator_factory = translator_factory
        self._tts_factory = tts_factory
        self._voice = voice
        self._fmt = fmt
        self._gather = gather

    async def source_hash(self, source_kind: str, source_id: int) -> str | None:
        """The current fingerprint of a translatable source, or ``None`` if it is
        gone. Workers fence a delayed job with this narrow read.
        """
        text = await self._assets.resolve_source_text(source_kind, source_id)
        return None if text is None else content_hash(text)

    async def translate(self, source_kind: str, source_id: int, lang: str) -> str:
        """Translate a source into ``lang``, cache-first and hash-verified.

        A repeat call with unchanged source spends nothing. Empty or whitespace
        source returns as-is: nothing to translate, no row worth writing.
        """
        text = await self._resolve_or_raise(source_kind, source_id)
        if not text.strip():
            return text
        params = normalize_asset_params("translate", lang=lang)
        cached = await self._assets.get(source_kind, source_id, "translate", params, text)
        if cached is not None and cached.text_value is not None:
            return cached.text_value
        translator = self._translator_factory()
        if translator is None:
            raise ValueError("no LLM configured for translation")
        result = await translator.translate(text, lang)
        stored = await self._assets.put_text(
            source_kind, source_id, "translate", params, text, result
        )
        return stored.text_value or result

    async def translate_many(
        self, refs: Sequence[tuple[str, int]], lang: str, *, concurrency: int = 5
    ) -> list[BatchResult]:
        async def _one(ref: tuple[str, int]) -> str:
            return await self.translate(ref[0], ref[1], lang)

        return await self._gather(list(refs), _one, concurrency=concurrency)

    async def speak(
        self,
        source_kind: str,
        source_id: int,
        voice: str | None = None,
        fmt: str | None = None,
    ) -> Asset:
        """Synthesize speech for a source, cache-first and hash-verified.

        A verified hit never calls the provider. On a miss the provider runs, and the
        stub raises rather than caching fake audio. Empty source short-circuits to a
        placeholder asset.
        """
        voice = voice if voice is not None else self._voice
        fmt = fmt if fmt is not None else self._fmt
        params = normalize_asset_params("tts", voice=voice, fmt=fmt)
        text = await self._resolve_or_raise(source_kind, source_id)
        if not text.strip():
            return Asset(source_kind=source_kind, source_id=source_id, kind="tts", params=params)
        cached = await self._assets.get(source_kind, source_id, "tts", params, text)
        if cached is not None:
            return cached
        data = await self._tts_factory().synthesize(text, voice, fmt)
        return await self._assets.put_file(
            source_kind, source_id, "tts", params, text, data, ext=fmt
        )

    async def speak_many(
        self,
        refs: Sequence[tuple[str, int]],
        voice: str | None = None,
        fmt: str | None = None,
        *,
        concurrency: int = 5,
    ) -> list[BatchResult]:
        """Batch speech. Two identical refs in ONE batch may both miss and
        synthesize; the content-addressed write dedups the row, so the cost is at
        worst one wasted call.
        """

        async def _one(ref: tuple[str, int]) -> Asset:
            return await self.speak(ref[0], ref[1], voice, fmt)

        return await self._gather(list(refs), _one, concurrency=concurrency)

    async def get(self, asset_id: int) -> Asset | None:
        return await self._assets.get_by_id(asset_id)

    async def list(
        self, *, kind: str | None = None, limit: int | None = None, offset: int = 0
    ) -> list[Asset]:
        return await self._assets.list(kind=kind, limit=limit, offset=offset)

    async def delete(self, asset_id: int) -> bool:
        return await self._assets.delete(asset_id)

    async def purge(self, *, kind: str | None = None) -> int:
        return await self._assets.purge(kind=kind)

    async def sweep_orphans(self, *, min_age_seconds: float = 3600.0) -> int:
        """Remove old unreferenced binary files left by failed cache writes."""
        return await self._assets.sweep_orphans(min_age_seconds=min_age_seconds)

    async def _resolve_or_raise(self, source_kind: str, source_id: int) -> str:
        text = await self._assets.resolve_source_text(source_kind, source_id)
        if text is None:
            raise ValueError(f"no source text for ({source_kind!r}, {source_id})")
        return text
