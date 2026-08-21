"""Reference-addressed asset cache repository (hash-verified).

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

The filesystem primitives (staged writes, verified unlinks, orphan sweep) live in
`asset_file_store.py`; this module owns transactions, identity upserts, and the
coordination between the two.
"""

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import delete, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lexi_ai.constants import (
    SOURCE_KINDS,
)
from lexi_ai.db import session_scope
from lexi_ai.domain.asset_identity import content_hash
from lexi_ai.infrastructure.db.asset_file_store import AssetFileStore
from lexi_ai.infrastructure.db.models import Asset as AssetRow
from lexi_ai.infrastructure.db.models import Collocation, Example, Sense
from lexi_ai.read_models import Asset

logger = logging.getLogger(__name__)

# source_kind -> (ORM model, text column). Driven by SOURCE_KINDS so a kind can
# never be half-wired: a test asserts every SOURCE_KINDS member has an entry.
_SOURCE_TABLES = {
    "sense_def": (Sense, Sense.definition),
    "example": (Example, Example.text),
    "collocation": (Collocation, Collocation.text),
}

_DEFAULT_ORPHAN_GRACE_SECONDS = 60 * 60


def _check_source_kind(source_kind: str) -> None:
    if source_kind not in SOURCE_KINDS:
        raise ValueError(f"unknown source kind: {source_kind!r}")


class AssetRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], cache_dir: str):
        self._session_factory = session_factory
        self._files = AssetFileStore(cache_dir)
        # Deferred-unlink listeners whose transaction has ended, awaiting removal.
        # See `_detach_spent_listeners` for why they are not removed in place.
        self._spent_listeners: list[tuple] = []
        # One repository may serve concurrent TTS calls and maintenance. Hold this
        # across the file write plus row transaction so a local sweep cannot observe
        # the deliberate write-before-row window as an old orphan.
        self._file_lock = asyncio.Lock()
        self._caller_file_locks: dict[object, object | None] = {}
        self._file_lock_listeners: list[tuple] = []

    @property
    def cache_dir(self) -> Path:
        """The configured cache directory (the file store's root)."""
        return self._files.root

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
            if row.file_path is not None and not self._files.exists(row.file_path):
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
        """Store a text asset while excluding local file-GC races."""
        self._detach_file_lock_listeners()
        async with self._file_lock:
            return await self._put_text(
                source_kind, source_id, kind, params, source_text, text_value, meta
            )

    async def _put_text(
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
                paths = await self._unreferenced_paths(
                    session,
                    [row.file_path],
                    exclude_ids=[row.id] if row.id is not None else (),
                )
                self._unlink_after_commit(session, paths)
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
                paths = await self._unreferenced_paths(
                    session,
                    [row.file_path],
                    exclude_ids=[row.id] if row.id is not None else (),
                )
                self._unlink_after_commit(session, paths)
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
        """Write one binary asset while excluding local sweep races."""
        self._detach_file_lock_listeners()
        async with self._file_lock:
            return await self._put_file(
                source_kind, source_id, kind, params, source_text, data, ext, meta
            )

    async def _put_file(
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
        prefix; see :meth:`AssetFileStore.relative_path` for the token sanitizing.
        Bytes are staged and fsynced first. An unreferenced final path is atomically
        installed before its row; a path already referenced by the DB is replaced
        only after the row transaction commits, so a rollback never corrupts a live
        asset. A failed new-row write may leave an inert orphan;
        :meth:`sweep_orphans` removes old unreferenced files later."""
        _check_source_kind(source_kind)
        h = content_hash(source_text)
        rel_path = self._files.relative_path(h, params, ext)
        temporary = self._files.stage(rel_path, data)
        try:
            async with session_scope(self._session_factory) as session:
                path_is_referenced = await self._path_is_referenced(session, rel_path)
                replace_after_commit = path_is_referenced and self._files.exists(rel_path)
                if not replace_after_commit:
                    self._files.replace(temporary, rel_path)
                    temporary = None
                row = await self._get(session, source_kind, source_id, kind, params)
                if row is not None:
                    if row.file_path is not None and row.file_path != rel_path:
                        paths = await self._unreferenced_paths(
                            session,
                            [row.file_path],
                            exclude_ids=[row.id] if row.id is not None else (),
                        )
                        self._unlink_after_commit(session, paths)
                    row.content_hash = h
                    row.file_path = rel_path
                    row.text_value = None
                    row.meta = meta
                    await session.flush()
                else:
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
                            paths = await self._unreferenced_paths(
                                session,
                                [row.file_path],
                                exclude_ids=[row.id] if row.id is not None else (),
                            )
                            self._unlink_after_commit(session, paths)
                        row.content_hash = h
                        row.file_path = rel_path
                        row.text_value = None
                        row.meta = meta
                        await session.flush()
                stored = self._to_asset(row)
            if replace_after_commit:
                try:
                    self._files.replace(temporary, rel_path)
                except OSError:
                    # The committed row still has the previous complete file. Keep
                    # the staged candidate for a later retry/sweep rather than
                    # turning a cache-maintenance failure into a request failure.
                    logger.warning(
                        "Could not install staged asset file after commit",
                        extra={"file_path": rel_path},
                        exc_info=True,
                    )
                    temporary = None
                else:
                    temporary = None
            return stored
        finally:
            if temporary is not None:
                self._files.discard(temporary)

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

    @staticmethod
    async def _path_is_referenced(session: AsyncSession, file_path: str) -> bool:
        result = await session.execute(
            select(AssetRow.id).where(AssetRow.file_path == file_path).limit(1)
        )
        return result.scalar_one_or_none() is not None

    async def _unreferenced_paths(
        self,
        session: AsyncSession,
        file_paths: Sequence[str | None],
        *,
        exclude_ids: Sequence[int] = (),
    ) -> list[str]:
        """Return paths no remaining asset row references after this operation."""
        candidates = {path for path in file_paths if path is not None}
        if not candidates:
            return []

        stmt = select(AssetRow.file_path).where(AssetRow.file_path.in_(candidates))
        if exclude_ids:
            stmt = stmt.where(AssetRow.id.not_in(exclude_ids))
        shared = {
            path for path in (await session.execute(stmt)).scalars().all() if path is not None
        }
        return [path for path in candidates if path not in shared]

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
        owns_lock = await self._acquire_caller_file_lock(session)
        try:
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
            if owns_lock:
                self._bind_caller_file_lock(session)
            if not rows:
                return 0
            paths = await self._unreferenced_paths(
                session,
                [row.file_path for row in rows],
                exclude_ids=[row.id for row in rows if row.id is not None],
            )
            self._unlink_after_commit(session, paths)
            await session.execute(
                delete(AssetRow).where(
                    AssetRow.source_kind == source_kind,
                    AssetRow.source_id.in_(source_ids),
                )
            )
            return len(rows)
        except asyncio.CancelledError:
            if owns_lock:
                self._release_unbound_caller_file_lock(session)
            raise
        except Exception:
            if owns_lock:
                self._release_unbound_caller_file_lock(session)
            raise

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
        """Delete one cached asset while excluding local binary writes."""
        self._detach_file_lock_listeners()
        async with self._file_lock:
            return await self._delete(asset_id)

    async def _delete(self, asset_id: int) -> bool:
        """Delete an asset row by id and unlink its backing file (best-effort).

        The unlink is deferred to commit: a rollback after the file was removed
        would leave the row pointing at nothing. Returns whether a row was
        removed. A missing file is ignored."""
        async with session_scope(self._session_factory) as session:
            row = await session.get(AssetRow, asset_id)
            if row is None:
                return False
            paths = await self._unreferenced_paths(
                session,
                [row.file_path],
                exclude_ids=[row.id] if row.id is not None else (),
            )
            self._unlink_after_commit(session, paths)
            await session.delete(row)
            return True

    async def purge(self, kind: str | None = None) -> int:
        """Purge cached assets while excluding local binary writes."""
        self._detach_file_lock_listeners()
        async with self._file_lock:
            return await self._purge(kind)

    async def _purge(self, kind: str | None = None) -> int:
        """Delete every cached asset (optionally one ``kind``), unlinking files.

        Returns the number of rows removed. Files are unlinked best-effort, after
        the transaction commits."""
        async with session_scope(self._session_factory) as session:
            stmt = select(AssetRow)
            if kind is not None:
                stmt = stmt.where(AssetRow.kind == kind)
            rows = (await session.execute(stmt)).scalars().all()
            paths = await self._unreferenced_paths(
                session,
                [row.file_path for row in rows],
                exclude_ids=[row.id for row in rows if row.id is not None],
            )
            self._unlink_after_commit(session, paths)
            for row in rows:
                await session.delete(row)
            return len(rows)

    async def sweep_orphans(self, *, min_age_seconds: float = _DEFAULT_ORPHAN_GRACE_SECONDS) -> int:
        """Remove old unreferenced files while excluding local asset writes."""
        self._detach_file_lock_listeners()
        async with self._file_lock:
            return await self._sweep_orphans(min_age_seconds=min_age_seconds)

    async def _sweep_orphans(
        self, *, min_age_seconds: float = _DEFAULT_ORPHAN_GRACE_SECONDS
    ) -> int:
        """Remove old managed files that no asset row references.

        The database snapshot of referenced paths is taken BEFORE the filesystem
        scan; see :meth:`AssetFileStore.sweep` for why that ordering and the age
        gate make an in-flight writer safe. Returns the number of files removed.
        """
        async with session_scope(self._session_factory) as session:
            referenced = {
                path
                for path in (
                    await session.execute(
                        select(AssetRow.file_path).where(AssetRow.file_path.is_not(None))
                    )
                )
                .scalars()
                .all()
                if path is not None
            }
        return self._files.sweep(referenced, min_age_seconds=min_age_seconds)

    async def _acquire_caller_file_lock(self, session: AsyncSession) -> bool:
        """Hold the filesystem lock until a caller-owned transaction ends."""
        self._detach_file_lock_listeners()
        sync_session = session.sync_session
        if sync_session in self._caller_file_locks:
            return False
        await self._file_lock.acquire()
        self._caller_file_locks[sync_session] = None
        return True

    def _bind_caller_file_lock(self, session: AsyncSession) -> None:
        sync_session = session.sync_session
        target = sync_session.get_transaction()
        if target is None:
            self._release_unbound_caller_file_lock(session)
            return
        self._caller_file_locks[sync_session] = target

        def _on_transaction_end(_session, transaction) -> None:
            if transaction is not target:
                return
            self._caller_file_locks.pop(sync_session, None)
            self._file_lock.release()
            self._file_lock_listeners.append((sync_session, _on_transaction_end))

        event.listen(sync_session, "after_transaction_end", _on_transaction_end)

    def _release_unbound_caller_file_lock(self, session: AsyncSession) -> None:
        sync_session = session.sync_session
        if self._caller_file_locks.get(sync_session) is None:
            self._caller_file_locks.pop(sync_session, None)
            self._file_lock.release()

    def _detach_file_lock_listeners(self) -> None:
        while self._file_lock_listeners:
            sync_session, on_end = self._file_lock_listeners.pop()
            event.remove(sync_session, "after_transaction_end", on_end)

    def _unlink_after_commit(self, session: AsyncSession, file_paths: Sequence[str | None]) -> None:
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
        expected = {path: self._files.identity(path) for path in paths}

        sync_session = session.sync_session
        # Keep both the operation savepoint and its root. A savepoint release is
        # not durable until the root commits, while a savepoint rollback cancels
        # this one deletion even if the root later commits.
        root = sync_session.get_transaction()
        operation = sync_session.get_nested_transaction() or root
        if root is None:  # pragma: no cover - no active transaction to wait on
            for path in paths:
                self._files.unlink_if_unchanged(path, expected[path])
            return

        self._detach_spent_listeners(sync_session)
        root_committed = False
        operation_committed = operation is root

        def _on_commit(_session) -> None:
            nonlocal operation_committed, root_committed
            if _session.get_transaction() is root and _session.get_nested_transaction() is None:
                root_committed = True
            if operation is not root and _session.get_nested_transaction() is operation:
                operation_committed = True

        def _on_transaction_end(_session, transaction) -> None:
            if transaction is not root:
                return  # a savepoint ends before the root decides durability
            # Mark for removal instead of removing here: this runs inside the
            # dispatch loop over the very deque `event.remove` would mutate.
            self._spent_listeners.append((sync_session, _on_commit, _on_transaction_end))
            if not root_committed or not operation_committed:
                return  # a rollback at either level keeps the file and row
            for path in paths:
                self._files.unlink_if_unchanged(path, expected[path])

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
