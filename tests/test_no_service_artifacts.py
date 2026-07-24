"""Release guard: the distribution is a transport-free library."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_service_package_is_not_importable():
    assert importlib.util.find_spec("lexi_service") is None


def test_service_only_roots_are_absent():
    for relative_path in ("lexi_service", "proto", "alembic", "alembic.ini", "deploy"):
        assert not (ROOT / relative_path).exists(), f"service artifact remains: {relative_path}"
