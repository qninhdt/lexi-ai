"""Filesystem half of the reference-addressed asset cache.

Everything about the cache directory that a database connection cannot know
lives here: sharded path shaping from sanitized tokens, staged-and-fsynced
writes, identity-checked unlinks, and the orphan sweep.

All paths crossing this boundary are RELATIVE to the cache dir; only the
methods themselves join them onto the root.
"""

import logging
import math
import os
import re
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# A managed file is ``{hash}.{params}.{ext}`` under a two-hex shard dir; a staged
# write adds one more ``.tmp`` suffix. The sweep refuses anything else, so the
# cache directory is never treated as wholly owned.
_MANAGED_FILE = re.compile(r"(?P<hash>[0-9a-f]{64})\.[^./\\]+\.[^./\\]+")
_TEMP_FILE = re.compile(r"\.(?P<hash>[0-9a-f]{64})\.[^./\\]+\.[^./\\]+\.[^./\\]+\.tmp")


class AssetFileStore:
    """Staged writes, verified unlinks, and orphan sweeping over the cache dir."""

    def __init__(self, cache_dir: str | Path):
        self._root = Path(cache_dir)

    @property
    def root(self) -> Path:
        """The cache directory every relative path resolves against."""
        return self._root

    # --- paths --------------------------------------------------------------

    def relative_path(self, digest: str, params: str, ext: str) -> str:
        """Sharded relative path for one asset: ``{hash[:2]}/{hash}.{params}.{ext}``.

        ``params``/``ext`` are sanitized to path-safe tokens (no traversal via env
        ``voice``/``fmt``); params folded in so two assets differing only by params
        (e.g. same text/fmt, different TTS voice) map to distinct files.
        """
        safe_ext = "".join(c if c.isalnum() else "-" for c in ext.strip().lstrip(".")).lower()
        safe_ext = safe_ext.strip("-") or "bin"
        safe_params = "".join(c if c.isalnum() else "-" for c in params).strip("-") or "x"
        return f"{digest[:2]}/{digest}.{safe_params}.{safe_ext}"

    def absolute(self, relative: str) -> Path:
        return self._root / relative

    def exists(self, relative: str) -> bool:
        return (self._root / relative).exists()

    # --- writing ------------------------------------------------------------

    def stage(self, relative: str, data: bytes) -> Path:
        """Write bytes to a temp file NEXT TO the final path and fsync it.

        Staging in the destination directory makes the later install an atomic
        rename on the same filesystem.
        """
        path = self._root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            assert temporary is not None
            return temporary
        except BaseException:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise

    def replace(self, staged: Path, relative: str) -> None:
        """Atomically install a staged file at its final path."""
        os.replace(staged, self._root / relative)

    @staticmethod
    def discard(staged: Path | None) -> None:
        """Remove a staged file that will not be installed."""
        if staged is not None:
            staged.unlink(missing_ok=True)

    # --- deleting -----------------------------------------------------------

    def identity(self, relative: str) -> tuple[int, int, int, int] | None:
        try:
            return self._stat_identity((self._root / relative).stat())
        except FileNotFoundError:
            return None
        except OSError:
            logger.warning(
                "Could not inspect cached asset file",
                extra={"file_path": relative},
                exc_info=True,
            )
            return None

    def unlink_if_unchanged(
        self, relative: str, expected: tuple[int, int, int, int] | None
    ) -> bool:
        """Unlink ``relative`` only if it still has the identity seen earlier.

        The deferred unlinks capture the stat identity when they are scheduled;
        checking it again at removal time keeps a file that was rewritten in the
        meantime (a newer writer's live asset).
        """
        if expected is None or self.identity(relative) != expected:
            return False
        return self.unlink(relative)

    def unlink(self, relative: str | None) -> bool:
        """Remove a backing file under the cache dir, ignoring a missing file.

        Synchronous, and deliberately left that way. Offloading this and the write
        above with `asyncio.to_thread` was tried and reverted: every caller is
        either the after-commit listener, which SQLAlchemy dispatches synchronously
        with no loop to await on, or a path with a transaction open on this
        connection. The await yields the event loop, a sibling task then works on
        the same connection, and its savepoint is invalidated —
        `OperationalError: no such savepoint`, reproduced by two concurrent TTS
        writes in roughly one run in six against a suite that was 10-for-10 clean
        before the change.

        The blocking call is real but bounded: a local cache write of a few KB.
        Making it async needs per-connection isolation for these writes first, which
        is a design change rather than a `to_thread` wrapper.
        """
        if relative is None:
            return False
        try:
            (self._root / relative).unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "Could not remove cached asset file",
                extra={"file_path": relative},
                exc_info=True,
            )
            return False
        return True

    # --- GC scan ------------------------------------------------------------

    def sweep(self, referenced: set[str], *, min_age_seconds: float) -> int:
        """Remove old managed files absent from ``referenced``; returns the count.

        The age gate is the concurrency guard: a writer stages bytes before its
        row commits, so a fresh unreferenced path may belong to an in-flight
        writer. Callers take their database snapshot of referenced paths BEFORE
        calling this; writers that start after that snapshot update the file mtime
        and remain inside the grace window. Unknown files are ignored rather than
        treating the whole cache directory as owned.
        """
        if not math.isfinite(min_age_seconds) or min_age_seconds < 0:
            raise ValueError("min_age_seconds must be finite and non-negative")
        if not self._root.exists():
            return 0

        cutoff = time.time() - min_age_seconds
        removed = 0
        for path in self._root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self._root)
            if len(relative.parts) != 2:
                continue
            if not re.fullmatch(r"[0-9a-f]{2}", relative.parts[0]):
                continue
            managed = _MANAGED_FILE.fullmatch(relative.name)
            temporary = _TEMP_FILE.fullmatch(relative.name)
            if managed is None and temporary is None:
                continue
            match = managed or temporary
            if relative.parts[0] != match.group("hash")[:2]:
                continue
            try:
                stat = path.stat()
                if stat.st_mtime > cutoff:
                    continue
            except FileNotFoundError:
                continue
            except OSError:
                logger.warning(
                    "Could not inspect cached asset file during sweep",
                    extra={"file_path": relative.as_posix()},
                    exc_info=True,
                )
                continue
            if managed is not None and relative.as_posix() in referenced:
                continue
            if self.unlink_if_unchanged(relative.as_posix(), self._stat_identity(stat)):
                removed += 1
        return removed

    @staticmethod
    def _stat_identity(stat: os.stat_result) -> tuple[int, int, int, int]:
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns
