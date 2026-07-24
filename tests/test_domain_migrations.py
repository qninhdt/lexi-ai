"""Focused checks for the library-owned PostgreSQL migration surface."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import UniqueConstraint, inspect

from lexi_ai.config import Settings
from lexi_ai.db import create_engine, init_models
from lexi_ai.models import Question
from tests.conftest import PG_SCHEMA, requires_postgres

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


def test_question_type_revision_is_chained_and_rendered_offline():
    revision_path = MIGRATIONS / "versions" / "20260723_02_question_type_levels.py"
    revision = revision_path.read_text()
    upgrade = _alembic("-x", "schema=dictionary", "upgrade", "head", "--sql")

    assert 'revision = "20260723_02"' in revision
    assert 'down_revision = "20260722_01"' in revision
    assert "type_id" in upgrade
    assert "render_format" in upgrade
    assert "difficulty_level" in upgrade
    assert "interaction_mode" in upgrade
    assert "content_hash" in upgrade
    assert "uq_question_content" in upgrade


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



def _reflected_question_schema(connection):
    inspector = inspect(connection)
    columns = {
        column["name"]: column["nullable"]
        for column in inspector.get_columns("questions", schema=PG_SCHEMA)
    }
    primary_key = tuple(
        inspector.get_pk_constraint("questions", schema=PG_SCHEMA)["constrained_columns"]
    )
    foreign_keys = {
        (
            tuple(foreign_key["constrained_columns"]),
            foreign_key["referred_table"],
            tuple(foreign_key["referred_columns"]),
        )
        for foreign_key in inspector.get_foreign_keys("questions", schema=PG_SCHEMA)
    }
    unique_constraints = {
        constraint["name"]: tuple(constraint["column_names"])
        for constraint in inspector.get_unique_constraints("questions", schema=PG_SCHEMA)
    }
    return columns, primary_key, foreign_keys, unique_constraints


@requires_postgres
async def test_postgres_question_schema_matches_orm(pg_session_factory):
    """Execute the full migration chain and compare its question schema to the ORM."""
    async with pg_session_factory() as session:
        connection = await session.connection()
        actual = await connection.run_sync(_reflected_question_schema)

    expected_columns = {
        column.name: column.nullable for column in Question.__table__.columns
    }
    expected_primary_key = tuple(column.name for column in Question.__table__.primary_key)
    expected_foreign_keys = {
        (
            tuple(element.parent.name for element in constraint.elements),
            next(iter(constraint.elements)).column.table.name,
            tuple(element.column.name for element in constraint.elements),
        )
        for constraint in Question.__table__.foreign_key_constraints
    }
    expected_unique_constraints = {
        constraint.name: tuple(column.name for column in constraint.columns)
        for constraint in Question.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    }

    assert actual == (
        expected_columns,
        expected_primary_key,
        expected_foreign_keys,
        expected_unique_constraints,
    )
