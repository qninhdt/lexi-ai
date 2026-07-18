"""Hermetic checks for the deploy-time reference artifact bootstrap."""

from __future__ import annotations

import importlib.util
from hashlib import sha256
from pathlib import Path

import pytest


@pytest.fixture
def bootstrap_module():
    path = Path(__file__).parents[1] / "deploy" / "reference-bootstrap.py"
    spec = importlib.util.spec_from_file_location("reference_bootstrap", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int) -> bytes:
        chunk, self._payload = self._payload[:size], self._payload[size:]
        return chunk


def _configure(monkeypatch, target: Path, payload: bytes, *, url: str = "https://artifacts.test/data"):
    monkeypatch.setenv("LEXI_REFERENCE_DATASET_URL", url)
    monkeypatch.setenv("LEXI_REFERENCE_DATASET_SHA256", sha256(payload).hexdigest())
    monkeypatch.setenv("LEXI_REFERENCE_DATASET_PATH", str(target))


def test_reference_bootstrap_downloads_and_verifies_the_artifact(
    bootstrap_module, monkeypatch, tmp_path
):
    payload = b"immutable reference data"
    target = tmp_path / "reference" / "cambridge.db"
    _configure(monkeypatch, target, payload)
    monkeypatch.setattr(bootstrap_module, "urlopen", lambda *_args, **_kwargs: _Response(payload))

    bootstrap_module.main()

    assert target.read_bytes() == payload
    assert target.stat().st_mode & 0o777 == 0o444


def test_reference_bootstrap_reuses_a_matching_cached_artifact(
    bootstrap_module, monkeypatch, tmp_path
):
    payload = b"already cached"
    target = tmp_path / "cambridge.db"
    target.write_bytes(payload)
    target.chmod(0o600)
    _configure(monkeypatch, target, payload)
    monkeypatch.setattr(
        bootstrap_module,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("matching artifact must not be downloaded"),
    )

    bootstrap_module.main()

    assert target.stat().st_mode & 0o777 == 0o444


def test_reference_bootstrap_refuses_non_https_artifact_urls(
    bootstrap_module, monkeypatch, tmp_path
):
    _configure(monkeypatch, tmp_path / "cambridge.db", b"data", url="http://artifacts.test/data")

    with pytest.raises(ValueError, match="must use HTTPS"):
        bootstrap_module.main()
