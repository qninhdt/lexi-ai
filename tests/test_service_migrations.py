"""Offline migration smoke tests for the service-owned Alembic chain."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALEMBIC = Path(sys.executable).with_name("alembic")


def _alembic(*args: str) -> str:
    completed = subprocess.run(
        [str(ALEMBIC), *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return completed.stdout


def test_service_migrations_generate_upgrade_and_rollback_sql():
    upgrade = _alembic("upgrade", "head", "--sql")
    rollback = _alembic("downgrade", "head:base", "--sql")

    assert "CREATE TABLE service_jobs" in upgrade
    assert "ALTER TABLE words ADD COLUMN IF NOT EXISTS generation_epoch" in upgrade
    assert "DROP TABLE service_jobs" in rollback
    assert "DROP COLUMN" not in rollback
