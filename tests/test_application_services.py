from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import pytest

from lexi_ai.read_models import Entry, SearchResult, Stats
from lexi_service.application.commands import (
    ExecuteGenerate,
    ExecuteTranslation,
    JobReference,
    JobSubmission,
    RequestContext,
    SubmitGenerate,
)
from lexi_service.application.errors import ApplicationError, ErrorCode, to_public_error
from lexi_service.application.policies import ServicePolicy
from lexi_service.application.queries import GetJobQuery, LookupEntryQuery, SearchQuery
from lexi_service.application.services import ExecutionService, QueryService, SubmissionService
from lexi_service.identity import Principal
from lexi_service.ports import JobRecord
from lexi_service.runtime import ServiceRuntime

POLICY = ServicePolicy(
    max_request_bytes=65536,
    max_query_chars=64,
    max_idempotency_key_chars=64,
    max_page_size=100,
    max_batch_size=25,
    max_provider_concurrency=2,
    maximum_job_age=timedelta(hours=1),
    provider_attempt_timeout=timedelta(seconds=30),
    max_retries=2,
)
PRINCIPAL = Principal("service-a", "tenant-a")
CONTEXT = RequestContext("request-1", PRINCIPAL)


@dataclass
class FakeDataset:
    fingerprint: str = "dataset-v1"


class FakeClock:
    def now(self):
        return datetime(2030, 1, 1, tzinfo=timezone.utc)


class FakePublisher:
    def __init__(self):
        self.submissions = []

    async def publish(self, submission):
        self.submissions.append(submission)
        return JobReference("job-1")


class FakeReader:
    def __init__(self, record=None):
        self.record = record
        self.calls = 0

    async def get(self, job_id):
        self.calls += 1
        return self.record


class FakeLexicon:
    def __init__(self):
        self.calls = []
        self.entry = Entry("word", "word", None, None, "done", 1)

    async def search(self, query):
        self.calls.append(("search", query))
        return [SearchResult(query, None, cambridge_id=1)]

    async def get_entry(self, word_id):
        self.calls.append(("get_entry", word_id))
        return self.entry

    async def generate(self, target):
        self.calls.append(("generate", target))
        return self.entry

    async def translate_field(self, source_kind, source_id, lang):
        self.calls.append(("translate", source_kind, source_id, lang))
        return "dịch"

    async def stats(self):
        self.calls.append(("stats",))
        return Stats({}, 0, 0, 0, 0, 0, {}, 0)


class FakeGate:
    def __init__(self):
        self.entered = 0

    @asynccontextmanager
    async def acquire(self, _tenant):
        self.entered += 1
        yield


class FakeSourcePreconditions:
    def __init__(self, matches=True):
        self.matches_value = matches
        self.calls = []

    async def matches(self, source_kind, source_id, expected_hash):
        self.calls.append((source_kind, source_id, expected_hash))
        return self.matches_value


class FakeResources:
    def __init__(self):
        self.closed = 0

    async def close(self):
        self.closed += 1


def job_submission(
    owner=PRINCIPAL,
    fingerprint="dataset-v1",
    *,
    accepted_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
    maximum_age_seconds=3600,
    max_retries=2,
):
    return JobSubmission(
        "generate",
        "request-1",
        owner,
        "key",
        1,
        fingerprint,
        accepted_at,
        {"display": "cat", "cambridge_id": 1, "lexi_word_id": None},
        maximum_age_seconds,
        max_retries,
    )


@pytest.mark.asyncio
async def test_query_rejects_missing_identity_before_lexicon_call():
    lexicon = FakeLexicon()
    service = QueryService(lexicon, FakeReader(), POLICY)

    with pytest.raises(ApplicationError) as raised:
        await service.search(SearchQuery(RequestContext("request-1", None), "cat"))

    assert raised.value.error.code is ErrorCode.UNAUTHENTICATED
    assert lexicon.calls == []


@pytest.mark.asyncio
async def test_submission_is_owned_and_fingerprinted_without_executing_lexicon():
    publisher = FakePublisher()
    service = SubmissionService(publisher, FakeDataset(), POLICY, FakeClock())
    target = SearchResult("cat", None, cambridge_id=42)

    job = await service.submit_generate(
        SubmitGenerate(CONTEXT, target, "request-key", "dataset-v1")
    )

    assert job.job_id == "job-1"
    assert publisher.submissions[0].owner == PRINCIPAL
    assert publisher.submissions[0].payload["cambridge_id"] == 42


@pytest.mark.asyncio
async def test_lookup_hides_pending_entry():
    lexicon = FakeLexicon()
    lexicon.entry = Entry("word", "word", None, None, "pending", 1)
    service = QueryService(lexicon, FakeReader(), POLICY)

    with pytest.raises(ApplicationError) as raised:
        await service.lookup(LookupEntryQuery(CONTEXT, 1))

    assert raised.value.error.code is ErrorCode.NOT_FOUND


@pytest.mark.asyncio
async def test_job_read_does_not_reveal_another_principals_job():
    record = JobRecord(JobReference("job-1"), Principal("service-b", "tenant-b"), "generate")
    service = QueryService(FakeLexicon(), FakeReader(record), POLICY)

    with pytest.raises(ApplicationError) as raised:
        await service.get_job(GetJobQuery(CONTEXT, "job-1"))

    assert raised.value.error.code is ErrorCode.NOT_FOUND


@pytest.mark.asyncio
async def test_worker_execution_has_no_publisher_and_uses_provider_gate():
    lexicon = FakeLexicon()
    gate = FakeGate()
    service = ExecutionService(
        lexicon, FakeDataset(), gate, POLICY, FakeClock(), FakeSourcePreconditions()
    )

    result = await service.execute_generate(
        ExecuteGenerate(job_submission(), SearchResult("cat", None, cambridge_id=1), 1, 1)
    )

    assert result is lexicon.entry
    assert gate.entered == 1
    assert not hasattr(service, "_publisher")


@pytest.mark.asyncio
async def test_worker_rejects_mismatched_reference_dataset_before_provider_call():
    lexicon = FakeLexicon()
    service = ExecutionService(
        lexicon, FakeDataset(), FakeGate(), POLICY, FakeClock(), FakeSourcePreconditions()
    )

    with pytest.raises(ApplicationError) as raised:
        await service.execute_generate(
            ExecuteGenerate(
                job_submission(fingerprint="dataset-v2"),
                SearchResult("cat", None, cambridge_id=1),
                1,
                1,
            )
        )

    assert raised.value.error.code is ErrorCode.PRECONDITION_FAILED
    assert lexicon.calls == []


@pytest.mark.asyncio
async def test_worker_rejects_payload_that_differs_from_the_persisted_job():
    lexicon = FakeLexicon()
    service = ExecutionService(
        lexicon, FakeDataset(), FakeGate(), POLICY, FakeClock(), FakeSourcePreconditions()
    )

    with pytest.raises(ApplicationError) as raised:
        await service.execute_generate(
            ExecuteGenerate(job_submission(), SearchResult("dog", None, cambridge_id=1), 1, 1)
        )

    assert raised.value.error.code is ErrorCode.PRECONDITION_FAILED
    assert lexicon.calls == []


@pytest.mark.asyncio
async def test_worker_uses_the_persisted_job_age_and_retry_cap():
    lexicon = FakeLexicon()
    service = ExecutionService(
        lexicon, FakeDataset(), FakeGate(), POLICY, FakeClock(), FakeSourcePreconditions()
    )

    with pytest.raises(ApplicationError) as expired:
        await service.execute_generate(
            ExecuteGenerate(
                job_submission(accepted_at=datetime(2029, 1, 1, tzinfo=timezone.utc)),
                SearchResult("cat", None, cambridge_id=1),
                1,
                1,
            )
        )
    with pytest.raises(ApplicationError) as retried:
        await service.execute_generate(
            ExecuteGenerate(
                job_submission(max_retries=0), SearchResult("cat", None, cambridge_id=1), 1, 2
            )
        )

    assert expired.value.error.code is ErrorCode.DEADLINE_EXCEEDED
    assert retried.value.error.code is ErrorCode.PRECONDITION_FAILED
    assert lexicon.calls == []


@pytest.mark.asyncio
async def test_worker_rejects_an_unsupported_persisted_job_contract():
    lexicon = FakeLexicon()
    service = ExecutionService(
        lexicon, FakeDataset(), FakeGate(), POLICY, FakeClock(), FakeSourcePreconditions()
    )

    with pytest.raises(ApplicationError) as operation:
        await service.execute_generate(
            ExecuteGenerate(
                replace(job_submission(), operation="translate"),
                SearchResult("cat", None, cambridge_id=1),
                1,
                1,
            )
        )
    with pytest.raises(ApplicationError) as version:
        await service.execute_generate(
            ExecuteGenerate(
                replace(job_submission(), payload_version=2),
                SearchResult("cat", None, cambridge_id=1),
                1,
                1,
            )
        )

    assert operation.value.error.code is ErrorCode.PRECONDITION_FAILED
    assert version.value.error.code is ErrorCode.PRECONDITION_FAILED
    assert lexicon.calls == []


@pytest.mark.asyncio
async def test_worker_rejects_translation_fields_that_differ_from_the_job():
    lexicon = FakeLexicon()
    service = ExecutionService(
        lexicon, FakeDataset(), FakeGate(), POLICY, FakeClock(), FakeSourcePreconditions()
    )
    job = replace(
        job_submission(),
        operation="translate",
        payload={"source_kind": "sense_def", "source_id": 1, "language": "vi", "source_hash": "a"},
    )

    with pytest.raises(ApplicationError) as raised:
        await service.execute_translation(ExecuteTranslation(job, "sense_def", 1, "fr", "a", 1))

    assert raised.value.error.code is ErrorCode.PRECONDITION_FAILED
    assert lexicon.calls == []


@pytest.mark.asyncio
async def test_translation_requires_a_current_source_hash_before_provider_use():
    lexicon = FakeLexicon()
    preconditions = FakeSourcePreconditions(matches=False)
    service = ExecutionService(
        lexicon, FakeDataset(), FakeGate(), POLICY, FakeClock(), preconditions
    )
    job = replace(
        job_submission(),
        operation="translate",
        payload={"source_kind": "sense_def", "source_id": 1, "language": "vi", "source_hash": "a"},
    )

    with pytest.raises(ApplicationError) as raised:
        await service.execute_translation(ExecuteTranslation(job, "sense_def", 1, "vi", "a", 1))

    assert raised.value.error.code is ErrorCode.PRECONDITION_FAILED
    assert preconditions.calls == [("sense_def", 1, "a")]
    assert lexicon.calls == []


@pytest.mark.asyncio
async def test_submission_rejects_an_oversized_generation_target():
    publisher = FakePublisher()
    service = SubmissionService(publisher, FakeDataset(), POLICY, FakeClock())
    target = SearchResult("x" * (POLICY.max_query_chars + 1), None, cambridge_id=1)

    with pytest.raises(ApplicationError) as raised:
        await service.submit_generate(SubmitGenerate(CONTEXT, target, "request-key", "dataset-v1"))

    assert raised.value.error.code is ErrorCode.VALIDATION
    assert publisher.submissions == []


@pytest.mark.asyncio
async def test_submission_rejects_an_unsupported_payload_version():
    publisher = FakePublisher()
    service = SubmissionService(publisher, FakeDataset(), POLICY, FakeClock())

    with pytest.raises(ApplicationError) as raised:
        await service.submit_generate(
            SubmitGenerate(
                CONTEXT, SearchResult("cat", None, cambridge_id=1), "request-key", "dataset-v1", 2
            )
        )

    assert raised.value.error.code is ErrorCode.PRECONDITION_FAILED
    assert publisher.submissions == []


@pytest.mark.asyncio
async def test_provider_failures_are_mapped_to_safe_public_errors():
    class FailingLexicon(FakeLexicon):
        async def generate(self, target):
            raise RuntimeError("database password leaked")

    service = ExecutionService(
        FailingLexicon(), FakeDataset(), FakeGate(), POLICY, FakeClock(), FakeSourcePreconditions()
    )

    with pytest.raises(ApplicationError) as raised:
        await service.execute_generate(
            ExecuteGenerate(job_submission(), SearchResult("cat", None, cambridge_id=1), 1, 1)
        )

    assert raised.value.error.code is ErrorCode.INTERNAL
    assert "password" not in raised.value.error.message


def test_unexpected_errors_never_expose_diagnostics():
    error = to_public_error(RuntimeError("postgres password: secret"))

    assert error.code is ErrorCode.INTERNAL
    assert "secret" not in error.message


@pytest.mark.asyncio
async def test_runtime_closes_owned_resources_once():
    resources = FakeResources()
    lexicon = FakeLexicon()
    runtime = ServiceRuntime.compose(
        lexicon=lexicon,
        jobs=FakeReader(),
        publisher=FakePublisher(),
        dataset=FakeDataset(),
        policy=POLICY,
        clock=FakeClock(),
        resources=resources,
        source_preconditions=FakeSourcePreconditions(),
        provider_gate=FakeGate(),
    )

    await runtime.close()
    await runtime.close()

    assert resources.closed == 1
    assert runtime.queries._lexicon is lexicon
    assert runtime.executions._lexicon is lexicon
