"""Skip-guarded Postgres test harness for the opt-in dual-DB tier.

The default suite is SQLite-only and hermetic (the README contract). A whole
class of defects — NUL rejection, ``VARCHAR(n)`` length enforcement, tz-aware
datetime binding — is real *only* on Postgres and invisible to SQLite, which is
lax where asyncpg is strict. This module stands up a real-Postgres engine over
``LEXI_TEST_PG_URL`` so those failures can actually fire.

The tier skips cleanly when the driver is absent OR the URL is unset — mirroring
``test_references.py`` skipping when ``./data`` is gone. Both must SKIP, never
error: without the driver guard a fixture opening ``postgresql+asyncpg://``
would ``ModuleNotFoundError`` at connect time instead of skipping.

Install + run the tier:

    uv sync --extra postgres
    LEXI_TEST_PG_URL=postgresql+asyncpg://user:pass@localhost/lexi_test uv run pytest
"""

import importlib.util
import os

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from lexi_ai.db import create_session_factory, init_models
from lexi_ai.models import Base

PG_URL = os.environ.get("LEXI_TEST_PG_URL")

_HAS_ASYNCPG = importlib.util.find_spec("asyncpg") is not None

# Both an absent driver AND a missing URL skip the whole tier rather than
# erroring (Phase 0 guard). Apply as ``pytestmark = requires_postgres`` on any
# module that uses ``pg_session_factory``.
requires_postgres = pytest.mark.skipif(
    not (_HAS_ASYNCPG and PG_URL),
    reason=(
        "Postgres tier: needs the asyncpg driver (uv sync --extra postgres) "
        "and LEXI_TEST_PG_URL set to a disposable database"
    ),
)


@pytest.fixture
async def pg_session_factory():
    """A session factory over a real Postgres with a freshly built schema.

    Function-scoped and self-cleaning: drop everything, ``init_models``, yield,
    then drop again — so each test starts on a clean schema and a crashed run
    leaves no tables behind (Phase 0 teardown risk note). ``PG_URL`` is
    guaranteed present here because callers gate on ``requires_postgres``.
    """
    engine = create_async_engine(PG_URL, future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await init_models(engine)
    try:
        yield create_session_factory(engine)
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()
