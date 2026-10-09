"""Selected-ID, read-only Cambridge access from SQLite or imported PostgreSQL tables."""

import asyncio
import base64
import binascii
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

from lexi_ai.errors import InvalidHandleError, InvalidResourceError


@dataclass(frozen=True)
class SourceSense:
    id: int
    pos: str | None
    definition: str
    examples: list[str] = field(default_factory=list)
    cefr_level: str | None = None
    ipa_uk: str | None = None
    ipa_us: str | None = None
    phrase_title: str | None = None
    headword: str | None = None


@dataclass(frozen=True)
class SourceEntry:
    id: int
    slug: str
    display: str
    entry_type: str
    senses: list[SourceSense]
    alternatives: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class SourceHit:
    id: int
    display: str
    entry_type: str


def encode_reference_id(entry_id: int) -> str:
    if type(entry_id) is not int or entry_id <= 0:
        raise InvalidHandleError("invalid source entry")
    return "entry_" + base64.urlsafe_b64encode(f"entry:{entry_id}".encode()).decode().rstrip("=")


def decode_reference_id(reference_id: str) -> int:
    if not isinstance(reference_id, str) or not reference_id.startswith("entry_"):
        raise InvalidHandleError("invalid reference ID")
    try:
        token = reference_id[6:]
        decoded = base64.b64decode(token + "=" * (-len(token) % 4), altchars=b"-_", validate=True)
        raw = decoded.decode("ascii")
        _, digits = raw.split(":")
        entry_id = int(digits)
        if raw != f"entry:{entry_id}" or encode_reference_id(entry_id) != reference_id:
            raise ValueError("noncanonical handle")
        return entry_id
    except (UnicodeError, ValueError, TypeError, binascii.Error) as exc:
        raise InvalidHandleError("invalid reference ID") from exc


_FETCH = """
SELECT w.id AS word_id, w.word, w.display_form, w.entry_type,
       s.id, s.definition, s.cefr_level, s.phrase_title, e.pos, e.headword,
       e.pronunciation_uk, e.pronunciation_us,
       (SELECT json_group_array(example) FROM
        (SELECT example FROM examples WHERE sense_id=s.id ORDER BY example_order,id)
        AS ordered_examples) AS examples,
       (SELECT json_group_array(json_array(alternative_word,alternative_type))
        FROM word_alternatives WHERE word_id=w.id) AS alternatives
FROM words w LEFT JOIN entries e ON e.word_id=w.id
LEFT JOIN senses s ON s.entry_id=e.id
WHERE w.id=:entry_id ORDER BY e.entry_order, s.sense_order, s.id
"""
_PROJECTION = """
SELECT w.id, w.word, COALESCE(w.display_form, w.word) AS display, w.entry_type,
       (SELECT json_group_array(alternative_word) FROM word_alternatives
        WHERE word_id=w.id) AS aliases, f.inflections
FROM words w LEFT JOIN (
    SELECT e.word_id, json_group_array(i.inflected_form) AS inflections
    FROM entry_inflections i JOIN entries e ON e.id=i.entry_id GROUP BY e.word_id
) f ON f.word_id=w.id
WHERE w.status='done'
AND EXISTS (SELECT 1 FROM entries e JOIN senses s ON s.entry_id=e.id WHERE e.word_id=w.id)
ORDER BY w.id
"""


def _array(value):
    return json.loads(value) if isinstance(value, str) else value or []


class Cambridge:
    def __init__(self, path: str | Path = "", *, db=None):
        self.path = Path(path) if path else None
        self.db = db

    def _connect(self) -> sqlite3.Connection:
        if self.path is None:
            from lexi_ai.datasets import download

            self.path = download("reference")
        if not self.path.is_file():
            raise InvalidResourceError("reference database unavailable")
        connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    async def _rows(self, sql, parameters):
        if self.db is not None:
            import re

            from sqlalchemy import text

            sql = sql.replace("json_group_array", "json_agg").replace(
                "json_array", "json_build_array"
            )
            sql = re.sub(
                r"\b(words|entries|senses|examples|word_alternatives|entry_inflections)\b",
                r"lexi_reference.\1",
                sql,
            )
            # Column aliases also use 'examples'; qualification only applies to tables.
            sql = sql.replace("AS lexi_reference.examples", "AS examples")
            async with self.db.read() as connection:
                return (await connection.execute(text(sql), parameters)).mappings().all()

        def read():
            with closing(self._connect()) as connection:
                return connection.execute(sql, parameters).fetchall()

        return await asyncio.to_thread(read)

    async def fetch_by_id(self, entry_id: int) -> SourceEntry | None:
        if type(entry_id) is not int or entry_id <= 0:
            raise InvalidHandleError("invalid source entry")
        rows = await self._rows(_FETCH, {"entry_id": entry_id})
        if not rows:
            return None
        row = rows[0]
        return SourceEntry(
            row["word_id"],
            row["word"],
            row["display_form"] or row["word"],
            row["entry_type"],
            [
                SourceSense(
                    id=item["id"],
                    pos=item["pos"],
                    definition=item["definition"],
                    cefr_level=item["cefr_level"],
                    phrase_title=item["phrase_title"],
                    headword=item["headword"],
                    ipa_uk=item["pronunciation_uk"],
                    ipa_us=item["pronunciation_us"],
                    examples=_array(item["examples"]),
                )
                for item in rows
                if item["id"] is not None
            ],
            [tuple(item) for item in _array(row["alternatives"])],
        )

    async def projection(self) -> list[tuple[SourceHit, list[str], list[str]]]:
        """Only searchable identity/surfaces; never hydrate the reference senses."""
        return [
            (
                SourceHit(item["id"], item["display"], item["entry_type"]),
                [item["word"], item["display"], *_array(item["aliases"])],
                _array(item["inflections"]),
            )
            for item in await self._rows(_PROJECTION, {})
        ]
