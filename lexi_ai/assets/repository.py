"""Content-addressed asset cache repository (Phase 4).

``content_hash`` + ``normalize_asset_params`` are the addressing contract: ONE
normalization on every call (read and write), like ``match_key``/``tag_key``. If
they diverge, the cache misses forever. ``put_file`` writes the file BEFORE the
row (a row implies a file); a missing file for an existing row is treated as a
miss and rewritten.
"""

import hashlib
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lexi_ai.constants import ASSET_KINDS, TRANSLATION_LANGUAGES
from lexi_ai.db import session_scope
from lexi_ai.models import Asset as AssetRow
from lexi_ai.normalize import _CTRL_RE
from lexi_ai.read_models import Asset


def content_hash(text: str) -> str:
    """sha256 hex of the NORMALIZED source text.

    Normalization (strip control chars, collapse whitespace, strip) runs before
    hashing so trailing/interior-whitespace variants of the same text collapse to
    one asset. This is the addressing contract — the SAME on every call.
    """
    s = _CTRL_RE.sub(" ", text)
    s = " ".join(s.split()).strip()
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def normalize_asset_params(kind: str, **kw: str | None) -> str:
    """Stable param token for an asset identity, normalized on read AND write.

    ``translate`` → a normalized lang code (``lang``); ``tts`` → ``voice|fmt``.
    Unknown kind → ``ValueError``.
    """
    if kind not in ASSET_KINDS:
        raise ValueError(f"unknown asset kind: {kind!r}")
    if kind == "translate":
        lang = _norm_token(kw.get("lang"))
        if lang not in TRANSLATION_LANGUAGES:
            raise ValueError(f"invalid/unsupported language code: {lang!r}")
        return lang
    # tts
    voice = _norm_token(kw.get("voice"))
    fmt = _norm_token(kw.get("fmt"))
    return f"{voice}|{fmt}"


def _norm_token(value: str | None) -> str:
    return (value or "").strip().lower()


class AssetRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], cache_dir: str):
        self._session_factory = session_factory
        self._cache_dir = Path(cache_dir)

    async def get(self, content_hash: str, kind: str, params: str) -> Asset | None:
        """FREE lookup by asset identity. Returns ``None`` on a miss OR when the
        row points at a file that no longer exists (defensive: caller rewrites)."""
        async with session_scope(self._session_factory) as session:
            row = await self._get(session, content_hash, kind, params)
            if row is None:
                return None
            if row.file_path is not None and not (self._cache_dir / row.file_path).exists():
                return None  # row without its file — treat as a miss
            return self._to_asset(row)

    async def put_text(
        self,
        content_hash: str,
        kind: str,
        params: str,
        text_value: str,
        meta: str | None = None,
    ) -> Asset:
        """Resolve-or-create a text asset under UNIQUE. Rejects embedded NUL (the
        questions-payload posture — round-trips safely on Postgres)."""
        if "\x00" in text_value:
            raise ValueError("text_value must not contain NUL")
        async with session_scope(self._session_factory) as session:
            existing = await self._get(session, content_hash, kind, params)
            if existing is not None:
                return self._to_asset(existing)
            row = AssetRow(
                content_hash=content_hash,
                kind=kind,
                params=params,
                text_value=text_value,
                meta=meta,
            )
            try:
                async with session.begin_nested():
                    session.add(row)
                    await session.flush()
            except IntegrityError:
                existing = await self._get(session, content_hash, kind, params)
                if existing is None:
                    raise
                return self._to_asset(existing)
            return self._to_asset(row)

    async def put_file(
        self,
        content_hash: str,
        kind: str,
        params: str,
        data: bytes,
        ext: str,
        meta: str | None = None,
    ) -> Asset:
        """Write bytes to a sharded path then resolve-or-create the row.

        Path: ``{cache_dir}/{hash[:2]}/{hash}.{ext}`` (shard by prefix to avoid one
        huge dir). The RELATIVE path is stored. File is written BEFORE the row so a
        row always implies a file. Idempotent: existing row → return as-is (the
        file was just (re)written, healing a prior row-without-file)."""
        safe_ext = ext.strip().lstrip(".").lower() or "bin"
        # Fold params into the filename: asset identity is (content_hash, kind,
        # params), so two assets differing only by params (e.g. same text/fmt,
        # different TTS voice) MUST map to distinct files — else the second row's
        # write silently overwrites the first's bytes and both serve one voice.
        safe_params = "".join(c if c.isalnum() else "-" for c in params).strip("-") or "x"
        rel_path = f"{content_hash[:2]}/{content_hash}.{safe_params}.{safe_ext}"
        abs_path = self._cache_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(data)
        async with session_scope(self._session_factory) as session:
            existing = await self._get(session, content_hash, kind, params)
            if existing is not None:
                return self._to_asset(existing)
            row = AssetRow(
                content_hash=content_hash,
                kind=kind,
                params=params,
                file_path=rel_path,
                meta=meta,
            )
            try:
                async with session.begin_nested():
                    session.add(row)
                    await session.flush()
            except IntegrityError:
                existing = await self._get(session, content_hash, kind, params)
                if existing is None:
                    raise
                return self._to_asset(existing)
            return self._to_asset(row)

    async def _get(
        self, session: AsyncSession, content_hash: str, kind: str, params: str
    ) -> AssetRow | None:
        result = await session.execute(
            select(AssetRow).where(
                AssetRow.content_hash == content_hash,
                AssetRow.kind == kind,
                AssetRow.params == params,
            )
        )
        return result.scalar_one_or_none()

    # --- management (inspect / delete) ------------------------------------

    async def get_by_id(self, asset_id: int) -> Asset | None:
        """A cached asset by its DB id, or ``None`` if absent. FREE."""
        async with session_scope(self._session_factory) as session:
            row = await session.get(AssetRow, asset_id)
            return self._to_asset(row) if row is not None else None

    async def list(
        self, kind: str | None = None, limit: int | None = None, offset: int = 0
    ) -> list[Asset]:
        """Cached assets, oldest id first, optionally filtered by ``kind``. FREE."""
        async with session_scope(self._session_factory) as session:
            stmt = select(AssetRow)
            if kind is not None:
                stmt = stmt.where(AssetRow.kind == kind)
            stmt = stmt.order_by(AssetRow.id).offset(offset)
            if limit is not None:
                stmt = stmt.limit(limit)
            rows = (await session.execute(stmt)).scalars().all()
            return [self._to_asset(r) for r in rows]

    async def delete(self, asset_id: int) -> bool:
        """Delete an asset row by id and unlink its backing file (best-effort).

        Returns whether a row was removed. A missing file is ignored (the row is
        still deleted) — the file may already be gone (e.g. cache pruned)."""
        async with session_scope(self._session_factory) as session:
            row = await session.get(AssetRow, asset_id)
            if row is None:
                return False
            self._unlink(row.file_path)
            await session.delete(row)
            return True

    async def purge(self, kind: str | None = None) -> int:
        """Delete every cached asset (optionally one ``kind``), unlinking files.

        Returns the number of rows removed. Files are unlinked best-effort."""
        async with session_scope(self._session_factory) as session:
            stmt = select(AssetRow)
            if kind is not None:
                stmt = stmt.where(AssetRow.kind == kind)
            rows = (await session.execute(stmt)).scalars().all()
            for row in rows:
                self._unlink(row.file_path)
                await session.delete(row)
            return len(rows)

    def _unlink(self, file_path: str | None) -> None:
        """Remove a backing file under the cache dir, ignoring a missing file."""
        if file_path is None:
            return
        (self._cache_dir / file_path).unlink(missing_ok=True)

    @staticmethod
    def _to_asset(row: AssetRow) -> Asset:
        return Asset(
            id=row.id,
            kind=row.kind,
            params=row.params,
            text_value=row.text_value,
            file_path=row.file_path,
            meta=row.meta,
        )
