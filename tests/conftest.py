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
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import text

from lexi_ai.config import Settings
from lexi_ai.db import create_engine, create_session_factory

# The hermetic tier uses the in-memory vector index. The production default is
# LanceDB, which would write a real on-disk store from every test that constructs
# a Lexicon — shared state across tests, and files in the working tree. Set at
# import time (this module loads first) because `get_settings()` reads the
# environment on every call, so anything constructed later sees it.
os.environ.setdefault("LEXI_VECTOR_BACKEND", "memory")

PG_URL = os.environ.get("LEXI_TEST_PG_URL")
ROOT = Path(__file__).resolve().parents[1]
DOMAIN_ALEMBIC_CONFIG = ROOT / "lexi_ai" / "migrations" / "alembic.ini"
PG_SCHEMA = "lexi_test"

_HAS_ASYNCPG = importlib.util.find_spec("asyncpg") is not None
_PG_READY = bool(_HAS_ASYNCPG and PG_URL)

# A tier that skips itself is invisible: the job goes green having proved
# nothing. Where the Postgres tier is supposed to run (CI, with a service
# container), LEXI_REQUIRE_PG=1 turns the skip into a collection error, so a
# dead container or a forgotten `--extra postgres` fails loudly instead of
# quietly shrinking the suite by 10 tests.
if os.environ.get("LEXI_REQUIRE_PG") == "1" and not _PG_READY:
    raise RuntimeError(
        "LEXI_REQUIRE_PG=1 but the Postgres tier cannot run: "
        f"asyncpg installed={_HAS_ASYNCPG}, LEXI_TEST_PG_URL set={bool(PG_URL)}"
    )

# Both an absent driver AND a missing URL skip the whole tier rather than
# erroring (Phase 0 guard). Apply as ``pytestmark = requires_postgres`` on any
# module that uses ``pg_session_factory``.
requires_postgres = pytest.mark.skipif(
    not _PG_READY,
    reason=(
        "Postgres tier: needs the asyncpg driver (uv sync --extra postgres) "
        "and LEXI_TEST_PG_URL set to a disposable database"
    ),
)

# Weaker gate for tests that only need the asyncpg DIALECT, not a server:
# SQLAlchemy imports the driver while building the engine, so `create_engine`
# on a postgresql+asyncpg URL raises ModuleNotFoundError on a base install even
# though nothing ever connects.
requires_asyncpg_driver = pytest.mark.skipif(
    not _HAS_ASYNCPG,
    reason="needs the asyncpg driver (uv sync --extra postgres); no server required",
)


@pytest.fixture
async def pg_session_factory():
    """A session factory over a real Postgres with a fresh Alembic schema.

    Function-scoped and self-cleaning. ``PG_URL`` is guaranteed present here
    because callers gate on ``requires_postgres``.
    """
    subprocess.run(
        [
            str(Path(sys.executable).with_name("alembic")),
            "-c",
            str(DOMAIN_ALEMBIC_CONFIG),
            "-x",
            f"schema={PG_SCHEMA}",
            "upgrade",
            "head",
        ],
        cwd=ROOT,
        check=True,
        env={**os.environ, "LEXI_DB_URL": PG_URL},
    )
    engine = create_engine(Settings(db_url=PG_URL, db_schema=PG_SCHEMA))
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE TABLE lexi_test.words CASCADE"))
    try:
        yield create_session_factory(engine)
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA IF EXISTS lexi_test CASCADE"))
        await engine.dispose()
