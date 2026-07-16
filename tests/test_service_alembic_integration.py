"""Real-Postgres service migration lifecycle, opt-in via LEXI_TEST_PG_URL."""

import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy import text

from lexi_ai.db import session_scope
from tests.conftest import PG_URL, requires_postgres

pytestmark = requires_postgres

ROOT = Path(__file__).resolve().parents[1]
ALEMBIC = Path(sys.executable).with_name("alembic")


def _migrate(*args: str) -> None:
    environment = {**os.environ, "LEXI_SERVICE_DATABASE_URL": PG_URL}
    subprocess.run([str(ALEMBIC), *args], cwd=ROOT, env=environment, check=True)


async def test_service_alembic_upgrade_and_rollback_preserve_library_fence(pg_session_factory):
    _migrate("upgrade", "head")
    async with session_scope(pg_session_factory) as session:
        service_jobs = await session.scalar(text("SELECT to_regclass('public.service_jobs')"))
        assert service_jobs == "service_jobs"

    _migrate("downgrade", "base")
    async with session_scope(pg_session_factory) as session:
        assert await session.scalar(text("SELECT to_regclass('public.service_jobs')")) is None
        assert (
            await session.scalar(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'words' AND column_name = 'generation_epoch'"
                )
            )
            == "generation_epoch"
        )
