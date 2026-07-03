"""Question persistence — CRUD for questions a plugin chose to save.

Mirrors the patterns of :mod:`lexi_ai.persistence.repository`: all writes go
through ``session_scope`` (commit/rollback), reads return detached column-only
data, and the JSON (de)serialization of ``payload`` happens ONLY at this
boundary — the rest of the system deals in dicts/dataclasses.

This class satisfies the ``QuestionStore`` protocol, so a plugin receives exactly
this CRUD surface via ``ctx.store`` (never a raw session). There is no dedup key
(a Phase-1 decision): ``insert`` always creates a new row.
"""

import json

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lexi_ai.db import session_scope
from lexi_ai.models import Question as QuestionRow
from lexi_ai.read_models import Question


class QuestionRepository:
    """CRUD for persisted questions. The ``QuestionStore`` handed to plugins."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    async def insert(self, q: Question) -> Question:
        """Persist a question; return it with its DB ``id`` set.

        Serializes ``payload`` to a JSON string at this boundary. Rejects a
        payload that serializes to text containing a NUL (``\\x00``) — Postgres
        text columns reject it, so we fail explicitly rather than corrupt on one
        backend and not the other.
        """
        payload_json = _dump_payload(q.payload)
        async with session_scope(self._session_factory) as session:
            row = QuestionRow(
                word_id=q.word_id,
                sense_id=q.sense_id,
                format=q.format,
                answer_kind=q.answer_kind,
                payload=payload_json,
            )
            session.add(row)
            await session.flush()
            new_id = row.id
        return Question(
            id=new_id,
            word_id=q.word_id,
            sense_id=q.sense_id,
            format=q.format,
            answer_kind=q.answer_kind,
            payload=q.payload,
        )

    async def list_for_word(self, word_id: int, fmt: str | None = None) -> list[Question]:
        """Persisted questions for a word, optionally filtered by ``format``."""
        async with session_scope(self._session_factory) as session:
            stmt = select(QuestionRow).where(QuestionRow.word_id == word_id)
            if fmt is not None:
                stmt = stmt.where(QuestionRow.format == fmt)
            stmt = stmt.order_by(QuestionRow.id)
            rows = (await session.execute(stmt)).scalars().all()
            return [_to_read_model(r) for r in rows]

    async def get(self, question_id: int) -> Question | None:
        """A single persisted question by id, or ``None`` if absent."""
        async with session_scope(self._session_factory) as session:
            row = await session.get(QuestionRow, question_id)
            return _to_read_model(row) if row is not None else None

    async def delete(self, question_id: int) -> bool:
        """Delete a question by id; return whether a row was removed."""
        async with session_scope(self._session_factory) as session:
            result = await session.execute(
                delete(QuestionRow).where(QuestionRow.id == question_id)
            )
            return (result.rowcount or 0) > 0


def _dump_payload(payload: dict) -> str:
    """JSON-encode a payload dict, rejecting embedded NUL in any string value.

    JSON escapes a NUL to ``\\u0000`` on the way out (so the stored text is
    Postgres-safe), but ``json.loads`` turns it back into a raw ``\\x00`` in the
    dict — which would then poison a downstream Postgres text write. Reject it at
    the boundary instead of storing a value that can't round-trip safely.
    """
    if _has_nul(payload):
        raise ValueError("question payload contains a NUL character (\\x00)")
    return json.dumps(payload, ensure_ascii=False)


def _has_nul(value: object) -> bool:
    """True if a NUL (``\\x00``) hides in any string within a JSON-like value."""
    if isinstance(value, str):
        return "\x00" in value
    if isinstance(value, dict):
        return any(_has_nul(k) or _has_nul(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_nul(v) for v in value)
    return False


def _to_read_model(row: QuestionRow) -> Question:
    """Build the public read model from a row, parsing the JSON payload."""
    return Question(
        id=row.id,
        word_id=row.word_id,
        sense_id=row.sense_id,
        format=row.format,
        answer_kind=row.answer_kind,
        payload=json.loads(row.payload),
    )
