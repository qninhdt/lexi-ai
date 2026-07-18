from types import SimpleNamespace

import pytest

from lexi_service.settings import ServiceSettings


def settings(**overrides):
    values = {
        "database_url": "postgresql+asyncpg://db/lexi",
        "redis_url": "redis://redis:6379/0",
        "reference_dataset_fingerprint": "dataset",
        "max_request_bytes": 1024,
        "max_query_chars": 64,
        "max_idempotency_key_chars": 64,
        "max_page_size": 10,
        "max_batch_size": 10,
        "max_provider_concurrency": 2,
        "maximum_job_age_seconds": 3600,
        "provider_attempt_timeout_seconds": 30,
        "max_retries": 2,
        "internal_service_token": "test-internal-token",
    }
    values.update(overrides)
    return ServiceSettings(**values)


def test_service_startup_requires_postgres():
    with pytest.raises(ValueError, match="PostgreSQL"):
        settings(database_url="sqlite+aiosqlite://").validate_startup()


def test_service_startup_accepts_valid_service_config():
    settings().validate_startup()


def test_service_startup_requires_redis():
    with pytest.raises(ValueError, match="Redis"):
        settings(redis_url="http://redis").validate_startup()


def test_service_startup_requires_internal_service_token():
    with pytest.raises(ValueError, match="internal service token"):
        settings(internal_service_token="").validate_startup()


def test_service_startup_requires_internal_service_subject():
    with pytest.raises(ValueError, match="internal service subject"):
        settings(internal_service_subject="").validate_startup()


def test_service_startup_rejects_a_plaintext_llm_url_with_an_api_key():
    library = SimpleNamespace(llm_api_key="secret", llm_base_url="http://provider.example/v1")
    with pytest.raises(ValueError, match="HTTPS LLM"):
        settings().validate_library_provider_settings(library)


def test_service_startup_allows_https_or_loopback_llm_providers():
    settings().validate_library_provider_settings(
        SimpleNamespace(llm_api_key="secret", llm_base_url="https://provider.example/v1")
    )
    settings().validate_library_provider_settings(
        SimpleNamespace(llm_api_key="secret", llm_base_url="http://127.0.0.1:8000/v1")
    )
