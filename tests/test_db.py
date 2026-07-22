"""SQLite bootstrap behavior for local Lexi development."""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from lexi_ai.db import init_models


async def test_init_models_creates_a_fresh_sqlite_schema():
    engine = create_async_engine("sqlite+aiosqlite://")
    try:
        await init_models(engine)
        async with engine.connect() as connection:
            tables = set(
                (
                    await connection.execute(
                        text("SELECT name FROM sqlite_master WHERE type = 'table'")
                    )
                ).scalars()
            )
        assert {"words", "senses", "questions"} <= tables
    finally:
        await engine.dispose()
