import pytest

from lexi_service.grpc_runner import _read_secret


def test_grpc_runner_requires_explicit_certificate_file(monkeypatch):
    monkeypatch.delenv("LEXI_SERVICE_GRPC_PRIVATE_KEY_FILE", raising=False)
    with pytest.raises(ValueError, match="GRPC_PRIVATE_KEY_FILE"):
        _read_secret("LEXI_SERVICE_GRPC_PRIVATE_KEY_FILE")


def test_grpc_runner_reads_credential_bytes_from_the_configured_file(monkeypatch, tmp_path):
    certificate = tmp_path / "client.key"
    certificate.write_bytes(b"private-key")
    monkeypatch.setenv("LEXI_SERVICE_GRPC_PRIVATE_KEY_FILE", str(certificate))
    assert _read_secret("LEXI_SERVICE_GRPC_PRIVATE_KEY_FILE") == b"private-key"
