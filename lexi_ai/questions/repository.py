"""Portable persistence for prepared assessment questions.

The public boundary speaks the answer-safe contract types, but persistence stays
UNCHANGED: each draft's flat ``payload`` is stored in the single ``payload``
column with the SAME canonical-json ``content_hash`` (so dedup / idempotency
identity never shifts). The internal :class:`PersistedQuestion` carrier bridges
the two — its ``render_kind`` maps to/from the stored ``render_format`` string.
"""

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
from lexi_ai.contracts.questions import RenderKind
from lexi_ai.db import session_scope
from lexi_ai.domain.questions import PersistedQuestion
from lexi_ai.infrastructure.db.models import Question as QuestionRow

_MAX_PAYLOAD_BYTES = 65_536


class QuestionRepository:
    """Question storage shared by all assessment plugins."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    async def insert(self, draft: PersistedQuestion) -> PersistedQuestion:
        """Insert once per canonical content identity and return the stored row.

        The pre-read is the common fast path. The savepoint catches a concurrent
        unique-key winner without aborting the outer transaction, which keeps the
        implementation portable across SQLite and Postgres.
        """
        _validate_contract(draft)
        payload_json = _dump_payload(draft.payload)
        content_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        async with session_scope(self._session_factory) as session:
            existing = await _find_existing(session, draft, content_hash)
            if existing is not None:
                return _to_persisted(existing)

            row = QuestionRow(
                word_id=draft.word_id,
                sense_id=draft.sense_id,
                type_id=draft.type_id,
                render_format=draft.render_kind.value,
                difficulty_level=draft.difficulty_level,
                interaction_mode=draft.interaction,
                payload=payload_json,
                content_hash=content_hash,
            )
            try:
                async with session.begin_nested():
                    session.add(row)
                    await session.flush()
            except IntegrityError:
                existing = await _find_existing(session, draft, content_hash)
                if existing is None:
                    raise
                return _to_persisted(existing)
            return _to_persisted(row)

    async def retrieve_one(
        self,
        sense_id: int,
        difficulty_level: int,
        type_id: str,
        excluded_ids: frozenset[int],
    ) -> PersistedQuestion | None:
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
            return _to_persisted(row) if row is not None else None

    async def list_for_word(
        self, word_id: int, type_id: str | None = None
    ) -> list[PersistedQuestion]:
        async with session_scope(self._session_factory) as session:
            stmt = select(QuestionRow).where(QuestionRow.word_id == word_id)
            if type_id is not None:
                stmt = stmt.where(QuestionRow.type_id == type_id)
            rows = (await session.execute(stmt.order_by(QuestionRow.id))).scalars().all()
            return [_to_persisted(row) for row in rows]

    async def list_for_sense(
        self, sense_id: int, type_id: str | None = None
    ) -> list[PersistedQuestion]:
        """List only rows bound to the requested sense, newest first."""
        async with session_scope(self._session_factory) as session:
            stmt = select(QuestionRow).where(QuestionRow.sense_id == sense_id)
            if type_id is not None:
                stmt = stmt.where(QuestionRow.type_id == type_id)
            rows = (
                await session.execute(stmt.order_by(QuestionRow.id.desc()))
            ).scalars().all()
            return [_to_persisted(row) for row in rows]

    async def get(self, question_id: int) -> PersistedQuestion | None:
        async with session_scope(self._session_factory) as session:
            row = await session.get(QuestionRow, question_id)
            return _to_persisted(row) if row is not None else None

    async def delete(self, question_id: int) -> bool:
        async with session_scope(self._session_factory) as session:
            result = await session.execute(
                delete(QuestionRow).where(QuestionRow.id == question_id)
            )
            return (result.rowcount or 0) > 0


async def _find_existing(
    session: AsyncSession, draft: PersistedQuestion, content_hash: str
) -> QuestionRow | None:
    stmt = select(QuestionRow).where(
        QuestionRow.word_id == draft.word_id,
        QuestionRow.sense_id == draft.sense_id,
        QuestionRow.type_id == draft.type_id,
        QuestionRow.difficulty_level == draft.difficulty_level,
        QuestionRow.content_hash == content_hash,
    )
    return (await session.execute(stmt)).scalar_one_or_none()


def _validate_contract(draft: PersistedQuestion) -> None:
    vocabularies = (
        ("type_id", draft.type_id, QUESTION_TYPES),
        ("render_format", draft.render_kind.value, RENDER_FORMATS),
        ("difficulty_level", draft.difficulty_level, DIFFICULTY_LEVELS),
        ("interaction_mode", draft.interaction, INTERACTION_MODES),
    )
    for field, value, vocabulary in vocabularies:
        if value not in vocabulary:
            raise ValueError(f"question {field} is out of vocabulary: {value!r}")
    if draft.interaction == "assessment" and draft.sense_id is None:
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


def _to_persisted(row: QuestionRow) -> PersistedQuestion:
    return PersistedQuestion(
        question_id=row.id,
        word_id=row.word_id,
        sense_id=row.sense_id,
        type_id=row.type_id,
        render_kind=RenderKind(row.render_format),
        difficulty_level=row.difficulty_level,
        interaction=row.interaction_mode,
        payload=json.loads(row.payload),
    )
