"""Portable persistence for prepared assessment questions."""

import hashlib
import json

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lexi_ai.constants import (
    DIFFICULTY_LEVELS,
    INTERACTION_MODES,
    QUESTION_TYPES,
    RENDER_FORMATS,
)
from lexi_ai.db import session_scope
from lexi_ai.models import Question as QuestionRow
from lexi_ai.read_models import Question

_MAX_PAYLOAD_BYTES = 65_536


class QuestionRepository:
    """Question storage shared by all assessment plugins."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    async def insert(self, question: Question) -> Question:
        """Insert once per canonical content identity and return the stored row.

        The pre-read is the common fast path. The savepoint catches a concurrent
        unique-key winner without aborting the outer transaction, which keeps the
        implementation portable across SQLite and Postgres.
        """
        _validate_contract(question)
        payload_json = _dump_payload(question.payload)
        content_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        async with session_scope(self._session_factory) as session:
            existing = await _find_existing(session, question, content_hash)
            if existing is not None:
                return _to_read_model(existing)

            row = QuestionRow(
                word_id=question.word_id,
                sense_id=question.sense_id,
                type_id=question.type_id,
                render_format=question.render_format,
                difficulty_level=question.difficulty_level,
                interaction_mode=question.interaction_mode,
                payload=payload_json,
                content_hash=content_hash,
            )
            try:
                async with session.begin_nested():
                    session.add(row)
                    await session.flush()
            except IntegrityError:
                existing = await _find_existing(session, question, content_hash)
                if existing is None:
                    raise
                return _to_read_model(existing)
            return _to_read_model(row)

    async def retrieve_one(
        self,
        sense_id: int,
        difficulty_level: int,
        type_id: str,
        excluded_ids: frozenset[int],
    ) -> Question | None:
        """Return the first exact unexcluded row.

        This is intentionally non-atomic: it does not claim or lock the row.
        Session-level duplicate avoidance belongs to the caller maintaining
        ``excluded_ids``.
        """
        async with session_scope(self._session_factory) as session:
            stmt = select(QuestionRow).where(
                QuestionRow.sense_id == sense_id,
                QuestionRow.difficulty_level == difficulty_level,
                QuestionRow.type_id == type_id,
            )
            if excluded_ids:
                stmt = stmt.where(QuestionRow.id.not_in(excluded_ids))
            result = await session.execute(
                stmt.order_by(QuestionRow.id).limit(1)
            )
            row = result.scalar_one_or_none()
            return _to_read_model(row) if row is not None else None

    async def list_for_word(
        self, word_id: int, type_id: str | None = None
    ) -> list[Question]:
        async with session_scope(self._session_factory) as session:
            stmt = select(QuestionRow).where(QuestionRow.word_id == word_id)
            if type_id is not None:
                stmt = stmt.where(QuestionRow.type_id == type_id)
            rows = (await session.execute(stmt.order_by(QuestionRow.id))).scalars().all()
            return [_to_read_model(row) for row in rows]

    async def list_for_sense(
        self, sense_id: int, type_id: str | None = None
    ) -> list[Question]:
        """List only rows bound to the requested sense, newest first."""
        async with session_scope(self._session_factory) as session:
            stmt = select(QuestionRow).where(QuestionRow.sense_id == sense_id)
            if type_id is not None:
                stmt = stmt.where(QuestionRow.type_id == type_id)
            rows = (
                await session.execute(stmt.order_by(QuestionRow.id.desc()))
            ).scalars().all()
            return [_to_read_model(row) for row in rows]

    async def get(self, question_id: int) -> Question | None:
        async with session_scope(self._session_factory) as session:
            row = await session.get(QuestionRow, question_id)
            return _to_read_model(row) if row is not None else None

    async def delete(self, question_id: int) -> bool:
        async with session_scope(self._session_factory) as session:
            result = await session.execute(
                delete(QuestionRow).where(QuestionRow.id == question_id)
            )
            return (result.rowcount or 0) > 0


async def _find_existing(
    session: AsyncSession, question: Question, content_hash: str
) -> QuestionRow | None:
    stmt = select(QuestionRow).where(
        QuestionRow.word_id == question.word_id,
        QuestionRow.sense_id == question.sense_id,
        QuestionRow.type_id == question.type_id,
        QuestionRow.difficulty_level == question.difficulty_level,
        QuestionRow.content_hash == content_hash,
    )
    return (await session.execute(stmt)).scalar_one_or_none()


def _validate_contract(question: Question) -> None:
    vocabularies = (
        ("type_id", question.type_id, QUESTION_TYPES),
        ("render_format", question.render_format, RENDER_FORMATS),
        ("difficulty_level", question.difficulty_level, DIFFICULTY_LEVELS),
        ("interaction_mode", question.interaction_mode, INTERACTION_MODES),
    )
    for field, value, vocabulary in vocabularies:
        if value not in vocabulary:
            raise ValueError(f"question {field} is out of vocabulary: {value!r}")
    if question.interaction_mode == "assessment" and question.sense_id is None:
        raise ValueError("assessment question requires a non-null sense_id")


def _dump_payload(payload: dict) -> str:
    """Return canonical JSON after recursive NUL and UTF-8 size checks."""
    if _has_nul(payload):
        raise ValueError("question payload contains a NUL character (\\x00)")
    dumped = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(dumped.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
        raise ValueError("question payload exceeds 65,536 UTF-8 bytes")
    return dumped


def _has_nul(value: object) -> bool:
    if isinstance(value, str):
        return "\x00" in value
    if isinstance(value, dict):
        return any(_has_nul(key) or _has_nul(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_nul(item) for item in value)
    return False


def _to_read_model(row: QuestionRow) -> Question:
    return Question(
        question_id=row.id,
        word_id=row.word_id,
        sense_id=row.sense_id,
        type_id=row.type_id,
        render_format=row.render_format,
        difficulty_level=row.difficulty_level,
        interaction_mode=row.interaction_mode,
        payload=json.loads(row.payload),
    )
