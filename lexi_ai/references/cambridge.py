"""Selected-ID, read-only Cambridge access over the external SQLite snapshot."""

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


def encode_available_id(entry_id: int) -> str:
    if type(entry_id) is not int or entry_id <= 0:
        raise InvalidHandleError("invalid source entry")
    return "entry_" + base64.urlsafe_b64encode(f"entry:{entry_id}".encode()).decode().rstrip("=")


def decode_available_id(available_id: str) -> int:
    if not isinstance(available_id, str) or not available_id.startswith("entry_"):
        raise InvalidHandleError("invalid available ID")
    try:
        token = available_id[6:]
        decoded = base64.b64decode(token + "=" * (-len(token) % 4), altchars=b"-_", validate=True)
        raw = decoded.decode("ascii")
        _, digits = raw.split(":")
        entry_id = int(digits)
        if raw != f"entry:{entry_id}" or encode_available_id(entry_id) != available_id:
            raise ValueError("noncanonical handle")
        return entry_id
    except (UnicodeError, ValueError, TypeError, binascii.Error) as exc:
        raise InvalidHandleError("invalid available ID") from exc


class Cambridge:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        if not self.path.is_file():
            raise InvalidResourceError("Cambridge database unavailable")
        connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def _fetch(self, entry_id: int) -> SourceEntry | None:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT w.id AS word_id, w.word, w.display_form, w.entry_type, "
                "s.id, s.definition, s.cefr_level, s.phrase_title, e.pos, "
                "e.pronunciation_uk, e.pronunciation_us, "
                "(SELECT json_group_array(example) FROM "
                " (SELECT example FROM examples WHERE sense_id=s.id ORDER BY example_order,id)) "
                "AS examples, "
                "(SELECT json_group_array(json_array(alternative_word,alternative_type)) "
                " FROM word_alternatives WHERE word_id=w.id) AS alternatives "
                "FROM words w LEFT JOIN entries e ON e.word_id=w.id "
                "LEFT JOIN senses s ON s.entry_id=e.id "
                "WHERE w.id = ? ORDER BY e.entry_order, s.sense_order, s.id",
                (entry_id,),
            ).fetchall()
            if not rows:
                return None
            row = rows[0]
            senses = [
                SourceSense(
                    id=item["id"],
                    pos=item["pos"],
                    definition=item["definition"],
                    cefr_level=item["cefr_level"],
                    phrase_title=item["phrase_title"],
                    ipa_uk=item["pronunciation_uk"],
                    ipa_us=item["pronunciation_us"],
                    examples=json.loads(item["examples"]),
                )
                for item in rows
                if item["id"] is not None
            ]
            alternatives = [tuple(item) for item in json.loads(row["alternatives"])]
            return SourceEntry(
                row["word_id"],
                row["word"],
                row["display_form"] or row["word"],
                row["entry_type"],
                senses,
                alternatives,
            )

    async def fetch_by_id(self, entry_id: int) -> SourceEntry | None:
        if type(entry_id) is not int or entry_id <= 0:
            raise InvalidHandleError("invalid source entry")
        return await asyncio.to_thread(self._fetch, entry_id)

    def _projection(self) -> list[tuple[SourceHit, list[str]]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT w.id, w.word, COALESCE(w.display_form, w.word) AS display, w.entry_type, "
                "(SELECT json_group_array(alternative_word) FROM word_alternatives "
                "WHERE word_id=w.id) AS aliases "
                "FROM words w WHERE w.status = 'done' "
                "AND EXISTS (SELECT 1 FROM entries e JOIN senses s ON s.entry_id = e.id "
                "WHERE e.word_id = w.id) ORDER BY w.id"
            ).fetchall()
            return [
                (
                    SourceHit(item["id"], item["display"], item["entry_type"]),
                    [item["word"], item["display"], *json.loads(item["aliases"])],
                )
                for item in rows
            ]

    async def projection(self) -> list[tuple[SourceHit, list[str]]]:
        """Only searchable identity/surfaces; never hydrate the reference senses."""
        return await asyncio.to_thread(self._projection)
