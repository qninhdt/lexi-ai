"""Text-addressed translation cache SQL; no source-row or file tracking."""

from sqlalchemy import delete, select

from lexi_ai.db.pagination import validate_page
from lexi_ai.models import Translation
from lexi_ai.schema import Translation as TranslationRow


def _dto(row: TranslationRow) -> Translation:
    return Translation(row.id, row.input_hash, row.target_language, row.content)


async def by_key(db, input_hash: str, language: str) -> Translation | None:
    async with db.read() as connection:
        record = (
            await connection.execute(
                select(TranslationRow.__table__).where(
                    TranslationRow.input_hash == input_hash,
                    TranslationRow.target_language == language,
                )
            )
        ).first()
        return _dto(record) if record else None


async def insert(db, input_hash: str, language: str, content: str) -> Translation:
    async with db.transaction() as session:
        record = TranslationRow(input_hash=input_hash, target_language=language, content=content)
        session.add(record)
        await session.flush()
        return _dto(record)


async def get(db, identifier: int) -> Translation | None:
    async with db.read() as connection:
        row = (
            await connection.execute(
                select(TranslationRow.__table__).where(TranslationRow.id == identifier)
            )
        ).first()
        return _dto(row) if row else None


async def list_rows(
    db, *, after_id: int | None = None, limit: int | None = None
) -> list[Translation]:
    validate_page(limit, after_id)
    statement = select(TranslationRow.__table__).order_by(TranslationRow.id).limit(limit)
    if after_id is not None:
        statement = statement.where(TranslationRow.id > after_id)
    async with db.read() as connection:
        return [_dto(row) for row in (await connection.execute(statement)).all()]


async def remove(db, identifier: int) -> bool:
    async with db.transaction() as session:
        result = await session.execute(
            delete(TranslationRow).where(TranslationRow.id == identifier)
        )
        return bool(result.rowcount)


async def purge(db) -> int:
    async with db.transaction() as session:
        result = await session.execute(delete(TranslationRow))
        return result.rowcount
