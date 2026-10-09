"""Download SQLite artifacts and import their existing identities into PostgreSQL."""

import asyncio
import hashlib
import os
import re
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError
from urllib.request import urlopen

from sqlalchemy import insert, select, text, tuple_
from sqlalchemy.exc import SQLAlchemyError

from lexi_ai.db.session import Database
from lexi_ai.errors import InvalidResourceError
from lexi_ai.questions.storage import _dto
from lexi_ai.references.schema import CAMBRIDGE_TABLES, datasets
from lexi_ai.schema import Base

REFERENCE_VERSION = "reference-data-v1"
REFERENCE_URL = "https://github.com/qninhdt/lexi-ai/releases/download/reference-data-v1/data"
REFERENCE_SHA256 = "147a99e1d723671a6488100d7742783c45fb1e254a515d834709442cb963b3a4"
CONTENT_VERSION = "content-data-v1"
CONTENT_URL = "https://github.com/qninhdt/lexi-ai/releases/download/content-data-v1/content.sqlite"


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(name: str) -> Path:
    if name not in {"reference", "content"}:
        raise ValueError("expected reference or content")
    version = REFERENCE_VERSION if name == "reference" else CONTENT_VERSION
    cache = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "lexi-ai" / version
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{name}.sqlite"
    digest_file = target.with_suffix(".sqlite.sha256")
    expected = REFERENCE_SHA256 if name == "reference" else None
    if expected is None and digest_file.is_file():
        expected = digest_file.read_text().strip()
    try:
        if expected is None:
            with urlopen(CONTENT_URL + ".sha256", timeout=60) as response:
                expected = response.read(256).decode().split()[0]
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise InvalidResourceError("invalid dataset checksum")
        if target.is_file() and checksum(target) == expected:
            return target
        url = REFERENCE_URL if name == "reference" else CONTENT_URL
        with tempfile.NamedTemporaryFile(dir=cache, suffix=".download", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with urlopen(url, timeout=60) as response:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                stream.close()
                if checksum(temporary) != expected:
                    raise InvalidResourceError("dataset checksum mismatch")
                temporary.replace(target)
                digest_file.write_text(expected + "\n")
            finally:
                temporary.unlink(missing_ok=True)
        return target
    except (OSError, URLError, UnicodeError, IndexError) as error:
        raise InvalidResourceError(
            f"Cannot download {name} dataset {version}; provide a local SQLite artifact"
        ) from error


async def _copy(session, source, table):
    columns = [c.name for c in table.c if c.name not in {"position", "type_position"}]
    keys = list(table.primary_key.columns)
    cursor = source.execute(
        f'SELECT {",".join(chr(34) + c + chr(34) for c in columns)} FROM "{table.name}" '
        f"ORDER BY {','.join(chr(34) + c.name + chr(34) for c in keys)}"
    )
    copied = 0
    while batch := cursor.fetchmany(500):
        rows = [dict(zip(columns, values, strict=True)) for values in batch]
        identifiers = [tuple(row[c.name] for c in keys) for row in rows]
        existing = {
            tuple(row[c.name] for c in keys): row
            for row in (
                await session.execute(select(table).where(tuple_(*keys).in_(identifiers)))
            ).mappings()
        }
        missing = []
        for identity, row in zip(identifiers, rows, strict=True):
            if table.name == "questions":
                try:
                    _dto(SimpleNamespace(**row))
                except (ValueError, TypeError, KeyError) as error:
                    raise InvalidResourceError(f"invalid question payload at {identity}") from error
            if identity not in existing:
                missing.append(row)
            elif any(existing[identity][c] != row[c] for c in columns):
                raise InvalidResourceError(f"dataset conflict in {table.name} at {identity}")
        if missing:
            await session.execute(insert(table), missing)
            copied += len(missing)
    return copied


async def import_dataset(db, name: str, path=None, *, questions=True) -> int:
    if not isinstance(db, Database) or db.engine.dialect.name != "postgresql":
        raise ValueError("dataset import requires an owned PostgreSQL database")
    async with db.read() as connection:
        prefix = (
            "reference"
            if name == "reference"
            else f"content.{await connection.scalar(text('SELECT current_schema()'))}"
        )
        installed = dict(
            (await connection.execute(select(datasets.c.name, datasets.c.checksum))).all()
        )
    parts = ["reference"] if name == "reference" else [f"{prefix}.words"]
    if name == "content" and questions:
        parts.append(f"{prefix}.questions")
    if name == "content" and "reference" not in installed:
        raise InvalidResourceError("import reference before importing content")
    if path is None and all(part in installed for part in parts):
        return 0
    path = Path(path) if path is not None else await asyncio.to_thread(download, name)
    if not path.is_file():
        raise InvalidResourceError("dataset file missing")
    wal = Path(str(path) + "-wal")
    if wal.is_file() and wal.stat().st_size:
        raise InvalidResourceError("dataset must be a standalone SQLite snapshot")
    digest = await asyncio.to_thread(checksum, path)
    copied = 0
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as source:
            source.execute("BEGIN")
            if source.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise InvalidResourceError("invalid SQLite dataset")
            if name == "content":
                from lexi_ai.migrations import inspect_head

                if source.execute("SELECT version_num FROM alembic_version").fetchall() != [
                    (inspect_head(),)
                ]:
                    raise InvalidResourceError("content schema version mismatch")
            async with db.transaction() as session:
                await session.execute(text("SELECT pg_advisory_xact_lock(1818589289)"))
                installed = dict(
                    (await session.execute(select(datasets.c.name, datasets.c.checksum))).all()
                )
                tables = (
                    list(CAMBRIDGE_TABLES)
                    if name == "reference"
                    else [
                        t
                        for t in Base.metadata.sorted_tables
                        if t.name != "translations" and (questions or t.name != "questions")
                    ]
                )
                # Hold writers until imported IDs and their sequences are consistent.
                await session.execute(
                    text(
                        "LOCK TABLE "
                        + ",".join(
                            f'"{t.schema}"."{t.name}"' if t.schema else f'"{t.name}"'
                            for t in tables
                        )
                        + " IN SHARE ROW EXCLUSIVE MODE"
                    )
                )
                touched = []
                for table in tables:
                    part = prefix
                    if name == "content":
                        part += ".questions" if table.name == "questions" else ".words"
                    if installed.get(part) != digest:
                        copied += await _copy(session, source, table)
                        touched.append(table)
                for table in touched:
                    if name != "content" or "id" not in table.c:
                        continue
                    sequence = await session.scalar(
                        text("SELECT pg_get_serial_sequence(:table, 'id')"), {"table": table.name}
                    )
                    if sequence:
                        await session.execute(
                            text(
                                f"SELECT setval(:sequence, GREATEST("
                                f"(SELECT last_value FROM {sequence}),"
                                f'(SELECT COALESCE(MAX(id), 1) FROM "{table.name}")), true)'
                            ),
                            {"sequence": sequence},
                        )
                for part in parts:
                    await session.execute(
                        text(
                            "INSERT INTO lexi_reference.datasets(name,checksum) "
                            "VALUES(:name,:checksum) "
                            "ON CONFLICT(name) DO UPDATE SET checksum=excluded.checksum"
                        ),
                        {"name": part, "checksum": digest},
                    )
    except (sqlite3.Error, SQLAlchemyError) as error:
        raise InvalidResourceError(
            "dataset import failed; incompatible schema or content conflict"
        ) from error
    return copied
