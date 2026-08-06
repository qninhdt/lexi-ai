"""Reference-addressed asset cache repository (Phase 1, hash-verified).

Identity is ``(source_kind, source_id, kind, params)`` — the source row an asset
derives from plus its kind and a normalized param token — so a consumer holding a
``sense_id`` looks its translation/audio up directly. ``content_hash`` is NOT the
identity; it is the sha256 of the source text at write time, VERIFIED on read: a
reused/regenerated ``source_id`` whose current text no longer matches yields a
MISS (regenerate + overwrite), never poisoned content. Cascade-on-delete is
best-effort GC (reclaim rows + files) — the read-time hash verify is the
correctness guarantee, not the FK cascade.

``put_*`` writes the file BEFORE the row (a row implies a file); a missing file
for an existing row is treated as a miss and rewritten. ``normalize_asset_params``
runs ONCE on every call (read and write), like ``match_key``/``tag_key``.
"""

import hashlib
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import delete, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lexi_ai.config import get_settings
from lexi_ai.constants import (
    ASSET_KINDS,
    SOURCE_KINDS,
    TRANSLATION_LANGUAGES,
    TTS_FORMATS,
    TTS_VOICES,
)
from lexi_ai.db import session_scope
from lexi_ai.infrastructure.db.models import Asset as AssetRow
from lexi_ai.infrastructure.db.models import Collocation, Example, Sense
from lexi_ai.normalize import _CTRL_RE
from lexi_ai.read_models import Asset

# source_kind -> (ORM model, text column). Driven by SOURCE_KINDS so a kind can
# never be half-wired: a test asserts every SOURCE_KINDS member has an entry.
_SOURCE_TABLES = {
    "sense_def": (Sense, Sense.definition),
    "example": (Example, Example.text),
    "collocation": (Collocation, Collocation.text),
}


def content_hash(text: str) -> str:
    """sha256 hex of the NORMALIZED source text (VERIFY function, not identity).

    Normalization (strip control chars, collapse whitespace, strip) runs before
    hashing so trailing/interior-whitespace variants of the same text collapse to
    one hash. The SAME normalization on every call — this is the verify contract.
    """
    s = _CTRL_RE.sub(" ", text)
    s = " ".join(s.split()).strip()
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def normalize_asset_params(kind: str, **kw: str | None) -> str:
    """Stable param token for an asset identity, normalized on read AND write.

    ``translate`` → a normalized lang code (``lang``); ``tts`` → ``voice|fmt``.
    Unknown kind → ``ValueError``.

    Every free param is validated against a closed vocab at this one choke point
    (like ``lang`` against ``TRANSLATION_LANGUAGES``): ``voice``/``fmt`` against
    ``TTS_VOICES``/``TTS_FORMATS``. This closes the filename-collision bug where
    two distinct DB rows (``en-US`` vs ``en_US``) squashed to the SAME on-disk
    path and served each other's bytes. A ``None`` voice/fmt resolves to the
    configured default (``alloy``/``mp3``) BEFORE validation, so a default TTS
    call never hard-rejects on the happy path.
    """
    if kind not in ASSET_KINDS:
        raise ValueError(f"unknown asset kind: {kind!r}")
    if kind == "translate":
        lang = _norm_token(kw.get("lang"))
        if lang not in TRANSLATION_LANGUAGES:
            raise ValueError(f"invalid/unsupported language code: {lang!r}")
        return lang
    # tts — resolve None to the configured default, then validate both params.
    settings = get_settings()
    voice = _norm_token(kw.get("voice") if kw.get("voice") is not None else settings.tts_voice)
    fmt = _norm_token(kw.get("fmt") if kw.get("fmt") is not None else settings.tts_format)
    if voice not in TTS_VOICES:
        raise ValueError(f"invalid/unsupported TTS voice: {voice!r}")
    if fmt not in TTS_FORMATS:
        raise ValueError(f"invalid/unsupported TTS format: {fmt!r}")
    return f"{voice}|{fmt}"


def _norm_token(value: str | None) -> str:
    return (value or "").strip().lower()


def _check_source_kind(source_kind: str) -> None:
    if source_kind not in SOURCE_KINDS:
        raise ValueError(f"unknown source kind: {source_kind!r}")


class AssetRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], cache_dir: str):
        self._session_factory = session_factory
        self._cache_dir = Path(cache_dir)
        # Deferred-unlink listeners whose transaction has ended, awaiting removal.
        # See `_detach_spent_listeners` for why they are not removed in place.
        self._spent_listeners: list[tuple] = []

    # --- source resolution -------------------------------------------------

    async def resolve_source_text(self, source_kind: str, source_id: int) -> str | None:
        """Current source text for ``(source_kind, source_id)``, or ``None`` if the
        row is gone. Opens its own read session."""
        _check_source_kind(source_kind)
        async with session_scope(self._session_factory) as session:
            return await self._resolve(session, source_kind, source_id)

    @staticmethod
    async def _resolve(session: AsyncSession, source_kind: str, source_id: int) -> str | None:
        model, column = _SOURCE_TABLES[source_kind]
        row = await session.execute(select(column).where(model.id == source_id))
        return row.scalar_one_or_none()

    # --- read (hash-verified) ---------------------------------------------

    async def get(
        self, source_kind: str, source_id: int, kind: str, params: str, source_text: str
    ) -> Asset | None:
        """FREE lookup by reference identity, VERIFIED against ``source_text``.

        Returns ``None`` on: no row, a ``content_hash`` mismatch (source changed /
        id reused → stale, MUST regenerate), or a row pointing at a vanished file.
        """
        _check_source_kind(source_kind)
        want = content_hash(source_text)
        async with session_scope(self._session_factory) as session:
            row = await self._get(session, source_kind, source_id, kind, params)
            if row is None or row.content_hash != want:
                return None
            if row.file_path is not None and not (self._cache_dir / row.file_path).exists():
                return None  # row without its file — treat as a miss
            return self._to_asset(row)

    # --- write (upsert-refresh on the reference identity) ------------------

    async def put_text(
        self,
        source_kind: str,
        source_id: int,
        kind: str,
        params: str,
        source_text: str,
        text_value: str,
        meta: str | None = None,
    ) -> Asset:
        """Upsert a text asset on the reference identity, refreshing the hash.

        A stale row (reused/regenerated source) is OVERWRITTEN with the new hash +
        value — self-heal on next access. Rejects embedded NUL in ``text_value``
        (round-trips safely on Postgres)."""
        _check_source_kind(source_kind)
        if "\x00" in text_value:
            raise ValueError("text_value must not contain NUL")
        h = content_hash(source_text)
        async with session_scope(self._session_factory) as session:
            row = await self._get(session, source_kind, source_id, kind, params)
            if row is not None:
                self._unlink(row.file_path)  # drop any prior backing file
                row.content_hash = h
                row.text_value = text_value
                row.file_path = None
                row.meta = meta
                await session.flush()
                return self._to_asset(row)
            row = AssetRow(
                source_kind=source_kind,
                source_id=source_id,
                kind=kind,
                params=params,
                content_hash=h,
                text_value=text_value,
                meta=meta,
            )
            try:
                async with session.begin_nested():
                    session.add(row)
                    await session.flush()
            except IntegrityError:
                # Concurrent insert won the UNIQUE race — reload + overwrite.
                row = await self._get(session, source_kind, source_id, kind, params)
                if row is None:
                    raise
                self._unlink(row.file_path)  # drop any prior backing file
                row.content_hash = h
                row.text_value = text_value
                row.file_path = None
                row.meta = meta
                await session.flush()
            return self._to_asset(row)

    async def put_file(
        self,
        source_kind: str,
        source_id: int,
        kind: str,
        params: str,
        source_text: str,
        data: bytes,
        ext: str,
        meta: str | None = None,
    ) -> Asset:
        """Write bytes to a sharded path then upsert the row on the reference identity.

        Path: ``{cache_dir}/{hash[:2]}/{hash}.{params}.{ext}`` — sharded by hash
        prefix, params folded in so two assets differing only by params (e.g. same
        text/fmt, different TTS voice) map to distinct files. ``params``/``ext`` are
        sanitized to path-safe tokens (no traversal via env ``voice``/``fmt``). The
        file is written BEFORE the row so a row always implies a file. A stale row is
        overwritten (self-heal). If the row write fails with a non-IntegrityError,
        the just-written file is unlinked (only when THIS call created it) so a
        failure does not orphan a file with no backing row."""
        _check_source_kind(source_kind)
        h = content_hash(source_text)
        safe_ext = "".join(c if c.isalnum() else "-" for c in ext.strip().lstrip(".")).lower()
        safe_ext = safe_ext.strip("-") or "bin"
        safe_params = "".join(c if c.isalnum() else "-" for c in params).strip("-") or "x"
        rel_path = f"{h[:2]}/{h}.{safe_params}.{safe_ext}"
        abs_path = self._cache_dir / rel_path
        # Did THIS call create the file? A pre-existing path (same content-addressed
        # rel_path from an earlier put) must never be unlinked on a failure below —
        # only a file we just wrote is ours to roll back.
        file_preexisted = abs_path.exists()
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(data)
        # The file is written BEFORE the row so a row always implies a file. If the
        # row write fails with a NON-IntegrityError (IntegrityError is handled below
        # as a concurrent-insert adopt), unlink the just-written file so a failure
        # does not orphan it on disk — but only when this call created it.
        try:
            async with session_scope(self._session_factory) as session:
                row = await self._get(session, source_kind, source_id, kind, params)
                if row is not None:
                    if row.file_path is not None and row.file_path != rel_path:
                        self._unlink(row.file_path)
                    row.content_hash = h
                    row.file_path = rel_path
                    row.text_value = None
                    row.meta = meta
                    await session.flush()
                    return self._to_asset(row)
                row = AssetRow(
                    source_kind=source_kind,
                    source_id=source_id,
                    kind=kind,
                    params=params,
                    content_hash=h,
                    file_path=rel_path,
                    meta=meta,
                )
                try:
                    async with session.begin_nested():
                        session.add(row)
                        await session.flush()
                except IntegrityError:
                    row = await self._get(session, source_kind, source_id, kind, params)
                    if row is None:
                        raise
                    if row.file_path is not None and row.file_path != rel_path:
                        self._unlink(row.file_path)
                    row.content_hash = h
                    row.file_path = rel_path
                    row.text_value = None
                    row.meta = meta
                    await session.flush()
                return self._to_asset(row)
        except IntegrityError:
            # Handled-then-reraised concurrent race (adopt found no row): leave the
            # file — the winning row's put owns an identical content-addressed path.
            raise
        except Exception:
            if not file_preexisted:
                self._unlink(rel_path)
            raise

    async def _get(
        self, session: AsyncSession, source_kind: str, source_id: int, kind: str, params: str
    ) -> AssetRow | None:
        result = await session.execute(
            select(AssetRow).where(
                AssetRow.source_kind == source_kind,
                AssetRow.source_id == source_id,
                AssetRow.kind == kind,
                AssetRow.params == params,
            )
        )
        return result.scalar_one_or_none()

    # --- best-effort GC (runs on the CALLER's session) --------------------

    async def delete_by_source(
        self, session: AsyncSession, source_kind: str, source_id: int
    ) -> int:
        """Delete every asset row for ``(source_kind, source_id)`` and unlink its
        file, using the CALLER's session so the deletes commit/roll back with the
        caller's transaction (never a self-opened scope — that breaks atomicity).

        Best-effort GC: a missed row is inert (read-time hash verify prevents a
        mis-serve). Returns the number of rows removed. Enumerates rows first so
        files can be unlinked (a bulk Core delete never exposes child paths)."""
        return await self.delete_by_source_ids(session, source_kind, [source_id])

    async def delete_by_source_ids(
        self, session: AsyncSession, source_kind: str, source_ids: list[int]
    ) -> int:
        """Bulk :meth:`delete_by_source` — one SELECT + one Core delete for the
        whole ``source_ids`` set, on the CALLER's session (atomic with its
        transaction). Enumerates first to unlink files (a bulk delete never
        exposes child paths), then issues a single ``delete(...).where(IN)``.
        Returns the number of rows removed. Empty ``source_ids`` → no-op."""
        _check_source_kind(source_kind)
        if not source_ids:
            return 0
        rows = (
            (
                await session.execute(
                    select(AssetRow).where(
                        AssetRow.source_kind == source_kind,
                        AssetRow.source_id.in_(source_ids),
                    )
                )
            )
            .scalars()
            .all()
        )
        if not rows:
            return 0
        self._unlink_after_commit(session, [row.file_path for row in rows])
        await session.execute(
            delete(AssetRow).where(
                AssetRow.source_kind == source_kind,
                AssetRow.source_id.in_(source_ids),
            )
        )
        return len(rows)

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

        The unlink is deferred to commit: a rollback after the file was removed
        would leave the row pointing at nothing. Returns whether a row was
        removed. A missing file is ignored."""
        async with session_scope(self._session_factory) as session:
            row = await session.get(AssetRow, asset_id)
            if row is None:
                return False
            self._unlink_after_commit(session, [row.file_path])
            await session.delete(row)
            return True

    async def purge(self, kind: str | None = None) -> int:
        """Delete every cached asset (optionally one ``kind``), unlinking files.

        Returns the number of rows removed. Files are unlinked best-effort, after
        the transaction commits."""
        async with session_scope(self._session_factory) as session:
            stmt = select(AssetRow)
            if kind is not None:
                stmt = stmt.where(AssetRow.kind == kind)
            rows = (await session.execute(stmt)).scalars().all()
            self._unlink_after_commit(session, [row.file_path for row in rows])
            for row in rows:
                await session.delete(row)
            return len(rows)

    def _unlink(self, file_path: str | None) -> None:
        """Remove a backing file under the cache dir, ignoring a missing file."""
        if file_path is None:
            return
        (self._cache_dir / file_path).unlink(missing_ok=True)

    def _unlink_after_commit(
        self, session: AsyncSession, file_paths: Sequence[str | None]
    ) -> None:
        """Unlink these files once — and only once — the caller's transaction commits.

        Deleting the file first is not recoverable. The row deletions that go with
        it live in the caller's transaction, which can still roll back: a word
        delete that hits a constraint later, or any error between here and the
        commit, leaves the rows intact and their files gone. Every subsequent read
        of those rows is then a miss against a file that no longer exists.

        Waiting for the commit inverts the failure into the harmless direction. If
        the process dies between commit and unlink, the files are merely orphaned —
        the rows are gone, nothing serves them, and they are inert bytes on disk.
        That is what "best-effort GC" is allowed to mean; destroying live content
        is not.

        Two things make this precise rather than merely deferred:

        * A commit is recorded by ``after_commit`` but the unlink is performed from
          ``after_transaction_end``, which fires for a rollback too. Waiting on
          ``after_commit`` alone would fire on whatever commits next, so on a reused
          session a later, unrelated commit would happily unlink files whose own
          transaction rolled back — the original bug through a different door.
        * The handlers compare transaction identity, and are unregistered by the
          next call rather than from inside the dispatch — removing a listener
          while SQLAlchemy is iterating its own listener deque raises. Sessions
          here are caller-supplied and long-lived (``collect_word_assets`` calls in
          three times per word), so listeners that are never cleaned up accumulate
          one set per delete for the life of the session.

        Nested transactions (savepoints) are ignored deliberately: releasing a
        SAVEPOINT is not durability, and unlinking there would destroy files that
        an outer rollback still owns.
        """
        paths = [path for path in file_paths if path is not None]
        if not paths:
            return

        sync_session = session.sync_session
        # The transaction these deletes belong to. Anything that ends a DIFFERENT
        # transaction is somebody else's business.
        target = sync_session.get_transaction()
        if target is None:  # pragma: no cover - no active transaction to wait on
            for path in paths:
                self._unlink(path)
            return

        self._detach_spent_listeners(sync_session)
        committed = False

        def _on_commit(_session) -> None:
            nonlocal committed
            if _session.get_transaction() is target:
                committed = True

        def _on_transaction_end(_session, transaction) -> None:
            if transaction is not target:
                return  # a savepoint, or an unrelated later transaction
            # Mark for removal instead of removing here: this runs inside the
            # dispatch loop over the very deque `event.remove` would mutate.
            self._spent_listeners.append((sync_session, _on_commit, _on_transaction_end))
            if not committed:
                return  # rolled back: the rows survive, so their files must too
            for path in paths:
                self._unlink(path)

        event.listen(sync_session, "after_commit", _on_commit)
        event.listen(sync_session, "after_transaction_end", _on_transaction_end)

    def _detach_spent_listeners(self, _session: object) -> None:
        """Unregister listeners whose transaction has already ended.

        Done on the way IN to the next deferral rather than from inside a handler,
        because ``event.remove`` cannot run while SQLAlchemy iterates the listener
        collection it would mutate.
        """
        while self._spent_listeners:
            sync_session, on_commit, on_end = self._spent_listeners.pop()
            event.remove(sync_session, "after_commit", on_commit)
            event.remove(sync_session, "after_transaction_end", on_end)

    @staticmethod
    def _to_asset(row: AssetRow) -> Asset:
        return Asset(
            id=row.id,
            source_kind=row.source_kind,
            source_id=row.source_id,
            kind=row.kind,
            params=row.params,
            text_value=row.text_value,
            file_path=row.file_path,
            meta=row.meta,
        )
