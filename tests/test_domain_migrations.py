"""Focused checks for the library-owned PostgreSQL migration surface."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from lexi_ai.config import Settings
from lexi_ai.db import create_engine, init_models

ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_CONFIG = ROOT / "lexi_ai" / "migrations" / "alembic.ini"
MIGRATIONS = ROOT / "lexi_ai" / "migrations"


def _alembic(*args: str) -> str:
    completed = subprocess.run(
        [str(Path(sys.executable).with_name("alembic")), "-c", str(ALEMBIC_CONFIG), *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_postgres_engine_translates_the_default_schema():
    engine = create_engine(Settings(db_url="postgresql+asyncpg://unused", db_schema="dictionary"))
    try:
        assert engine.sync_engine._execution_options["schema_translate_map"] == {None: "dictionary"}
    finally:
        engine.sync_engine.dispose()


def test_sqlite_engine_does_not_translate_the_default_schema():
    engine = create_engine(Settings(db_url="sqlite+aiosqlite://"))
    try:
        assert "schema_translate_map" not in engine.sync_engine._execution_options
    finally:
        engine.sync_engine.dispose()


async def test_init_models_rejects_postgres_bootstrap_ddl():
    engine = create_engine(Settings(db_url="postgresql+asyncpg://unused"))
    try:
        with pytest.raises(RuntimeError, match="SQLite"):
            await init_models(engine)
    finally:
        await engine.dispose()


def test_domain_migrations_render_the_configured_schema_offline():
    upgrade = _alembic("-x", "schema=dictionary", "upgrade", "head", "--sql")

    assert "CREATE SCHEMA IF NOT EXISTS dictionary" in upgrade
    assert "CREATE TABLE dictionary.alembic_version" in upgrade
    assert "CREATE TABLE dictionary.words" in upgrade
    assert upgrade.index("CREATE SCHEMA") < upgrade.index("CREATE TABLE dictionary.alembic_version")


def test_baseline_is_frozen_independent_of_orm_metadata():
    baseline = (MIGRATIONS / "versions" / "20260722_01_baseline.py").read_text()
    frozen_schema = (MIGRATIONS / "frozen_schema.py").read_text()

    assert "lexi_ai.models" not in baseline
    assert "Base.metadata" not in baseline
    assert "Base.metadata" not in frozen_schema


def test_online_migrations_use_a_committing_engine_transaction():
    env = ast.parse((MIGRATIONS / "env.py").read_text())
    runner = next(
        node
        for node in env.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_migrations_online"
    )
    transaction = next(node for node in ast.walk(runner) if isinstance(node, ast.AsyncWith))
    call = transaction.items[0].context_expr

    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Attribute)
    assert call.func.attr == "begin"
