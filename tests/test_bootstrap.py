from hashlib import sha256

from lexi_service.bootstrap import FileDataset, policy_from_settings
from lexi_service.settings import ServiceSettings


def test_bootstrap_maps_service_settings_to_runtime_policy():
    settings = ServiceSettings(
        database_url="postgresql+asyncpg://db/lexi",
        redis_url="redis://redis:6379/0",
        reference_dataset_fingerprint="dataset-v1",
        max_request_bytes=1024,
        max_query_chars=64,
        max_idempotency_key_chars=64,
        max_page_size=10,
        max_batch_size=10,
        max_provider_concurrency=2,
        maximum_job_age_seconds=60,
        provider_attempt_timeout_seconds=10,
        max_retries=1,
    )
    policy = policy_from_settings(settings)
    assert policy.max_request_bytes == 1024
    assert FileDataset("/does-not-exist", "dataset-v1").fingerprint == "dataset-v1"


async def test_reference_dataset_readiness_requires_the_expected_digest(tmp_path):
    path = tmp_path / "cambridge.db"
    path.write_bytes(b"immutable dataset")
    dataset = FileDataset(str(path), sha256(b"immutable dataset").hexdigest())
    assert await dataset.ready()

    path.write_bytes(b"changed dataset")
    assert not await dataset.ready()
