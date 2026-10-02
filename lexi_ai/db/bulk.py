"""Bounded multi-row inserts and safe primary-key correlation."""

from sqlalchemy import insert, select


async def insert_rows(session, model, rows):
    """Real multi-row statements with bounded parameter counts, on both dialects."""
    for start in range(0, len(rows), 500):
        await session.execute(insert(model.__table__).values(rows[start : start + 500]))


async def insert_identified_rows(session, model, rows):
    """Return IDs in input order without assuming native RETURNING order.

    SQLite callers must reserve the writer with BEGIN IMMEDIATE before this
    allocation. PostgreSQL retains sequence-generated IDs and ordered RETURNING.
    No sentinel columns, persistent counters or correlation tokens are needed.
    """
    if not rows:
        return []
    if session.bind.dialect.name == "sqlite":
        last_id = await session.scalar(select(model.id).order_by(model.id.desc()).limit(1))
        identifiers = list(range((last_id or 0) + 1, (last_id or 0) + 1 + len(rows)))
        await insert_rows(
            session,
            model,
            [
                dict(value, id=identifier)
                for identifier, value in zip(identifiers, rows, strict=True)
            ],
        )
        return identifiers
    return (
        await session.scalars(insert(model).returning(model.id, sort_by_parameter_order=True), rows)
    ).all()
