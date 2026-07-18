"""Application services shared by future gRPC and HTTP adapters."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import timedelta

from lexi_ai.constants import QUESTION_FORMATS
from lexi_ai.read_models import Entry
from lexi_service.application.commands import (
    ExecuteGenerate,
    ExecuteTranslation,
    GenerateQuestions,
    GradeQuestion,
    JobReference,
    JobSubmission,
    SubmitGenerate,
    SubmitTranslation,
)
from lexi_service.application.errors import (
    ApplicationError,
    ErrorCode,
    public_error,
    to_public_error,
)
from lexi_service.application.policies import ServicePolicy
from lexi_service.application.queries import (
    GetJobQuery,
    GetQuestionQuery,
    GetSensesQuery,
    ListQuestionsQuery,
    LookupEntryQuery,
    SearchQuery,
    StatsQuery,
)
from lexi_service.identity import Principal
from lexi_service.observability.logging import log_event
from lexi_service.ports import (
    Clock,
    JobPublisher,
    JobReader,
    LexiconPort,
    ProviderGate,
    ReferenceDataset,
    SourcePreconditionVerifier,
)

logger = logging.getLogger(__name__)

# Pycil schedules reviews per sense.  A matching question spans an entry's senses
# and therefore cannot be bound to one FSRS card; keep it library-only for now.
SERVICE_QUESTION_FORMATS = frozenset(QUESTION_FORMATS - {"matching"})


def _require_principal(principal: Principal | None) -> Principal:
    if principal is None or not principal.subject:
        raise public_error(ErrorCode.UNAUTHENTICATED, "Authenticated service identity is required.")
    return principal


def _validate_text(value: str, maximum: int, name: str) -> None:
    if not value or len(value) > maximum:
        raise public_error(ErrorCode.VALIDATION, f"{name} is invalid.")


async def _safe_call(operation):
    try:
        return await operation()
    except ApplicationError:
        raise
    except Exception as exc:
        raise ApplicationError(to_public_error(exc)) from exc


class QueryService:
    def __init__(self, lexicon: LexiconPort, jobs: JobReader, policy: ServicePolicy):
        self._lexicon = lexicon
        self._jobs = jobs
        self._policy = policy

    async def search(self, query: SearchQuery):
        _require_principal(query.context.principal)
        _validate_text(query.query, self._policy.max_query_chars, "query")
        return await _safe_call(lambda: self._lexicon.search(query.query))

    async def lookup(self, query: LookupEntryQuery) -> Entry:
        _require_principal(query.context.principal)
        if query.word_id <= 0:
            raise public_error(ErrorCode.VALIDATION, "word_id is invalid.")
        entry = await _safe_call(lambda: self._lexicon.get_entry(query.word_id))
        if entry.status != "done":
            raise public_error(ErrorCode.NOT_FOUND, "Entry was not found.")
        return entry

    async def get_senses(self, query: GetSensesQuery):
        _require_principal(query.context.principal)
        if not query.sense_ids or len(query.sense_ids) > self._policy.max_batch_size:
            raise public_error(ErrorCode.VALIDATION, "sense_ids is invalid.")
        if any(sense_id <= 0 for sense_id in query.sense_ids):
            raise public_error(ErrorCode.VALIDATION, "sense_ids is invalid.")
        return await _safe_call(lambda: self._lexicon.get_senses(query.sense_ids))

    async def get_question(self, query: GetQuestionQuery):
        _require_principal(query.context.principal)
        if query.question_id <= 0:
            raise public_error(ErrorCode.VALIDATION, "question_id is invalid.")
        question = await _safe_call(lambda: self._lexicon.get_question(query.question_id))
        if question is None or question.id is None or question.sense_id is None:
            raise public_error(ErrorCode.NOT_FOUND, "Question was not found.")
        return question

    async def list_questions(self, query: ListQuestionsQuery):
        _require_principal(query.context.principal)
        if query.sense_id <= 0 or (
            query.format is not None and query.format not in SERVICE_QUESTION_FORMATS
        ):
            raise public_error(ErrorCode.VALIDATION, "question query is invalid.")
        return await _safe_call(
            lambda: self._lexicon.list_questions_for_sense(query.sense_id, query.format)
        )

    async def generate_questions(self, command: GenerateQuestions):
        _require_principal(command.context.principal)
        if command.word_id <= 0 or command.sense_id <= 0:
            raise public_error(ErrorCode.VALIDATION, "question target is invalid.")
        if (
            not command.formats
            or len(command.formats) > self._policy.max_batch_size
            or command.count < 1
            or command.count > self._policy.max_batch_size
            or any(fmt not in SERVICE_QUESTION_FORMATS for fmt in command.formats)
        ):
            raise public_error(ErrorCode.VALIDATION, "question generation is invalid.")
        questions = await _safe_call(
            lambda: self._lexicon.generate_questions_for_sense(
                command.word_id, command.sense_id, list(command.formats), command.count
            )
        )
        # Defend the service contract even if a plugin regresses: only durable
        # questions for the requested sense can become learner-facing.
        return [
            question
            for question in questions
            if question.id is not None and question.sense_id == command.sense_id
        ]

    async def grade_question(self, command: GradeQuestion):
        _require_principal(command.context.principal)
        if command.question_id <= 0:
            raise public_error(ErrorCode.VALIDATION, "question_id is invalid.")
        question = await _safe_call(lambda: self._lexicon.get_question(command.question_id))
        if question is None or question.id is None or question.sense_id is None:
            raise public_error(ErrorCode.NOT_FOUND, "Question was not found.")
        _validate_answer(question.answer_kind, command.answer, self._policy.max_query_chars)
        # The raw payload remains inside Lexi.  The only service result is its
        # verdict; callers never send answer keys back for grading.
        score = await _safe_call(
            lambda: self._lexicon.grade_question(command.question_id, command.answer)
        )
        if score is None:
            raise public_error(ErrorCode.NOT_FOUND, "Question was not found.")
        return score

    async def stats(self, query: StatsQuery):
        _require_principal(query.context.principal)
        return await _safe_call(self._lexicon.stats)

    async def get_job(self, query: GetJobQuery):
        principal = _require_principal(query.context.principal)
        record = await _safe_call(lambda: self._jobs.get(query.job_id))
        if record is None or record.owner.ownership_key != principal.ownership_key:
            raise public_error(ErrorCode.NOT_FOUND, "Job was not found.")
        return record


def _validate_answer(answer_kind: str, answer: object, max_chars: int) -> None:
    """Reject malformed/oversized answers before an LLM grader sees them."""
    if answer_kind == "single_choice":
        valid = isinstance(answer, (int, str)) and not isinstance(answer, bool)
    elif answer_kind in {"text_span", "free_text"}:
        valid = isinstance(answer, str) and 0 < len(answer) <= max_chars
    elif answer_kind == "matching":
        valid = isinstance(answer, list) and len(answer) <= 64 and all(
            isinstance(item, int) and not isinstance(item, bool) for item in answer
        )
    else:
        valid = False
    if not valid:
        raise public_error(ErrorCode.VALIDATION, "answer is invalid.")


class SubmissionService:
    """API-facing commands. This is the only service that can publish jobs."""

    def __init__(
        self,
        publisher: JobPublisher,
        dataset: ReferenceDataset,
        policy: ServicePolicy,
        clock: Clock,
    ):
        self._publisher = publisher
        self._dataset = dataset
        self._policy = policy
        self._clock = clock

    async def submit_generate(self, command: SubmitGenerate) -> JobReference:
        principal = self._validate_submission(
            command.context.principal,
            command.context.deadline,
            command.idempotency_key,
            command.reference_dataset_fingerprint,
            command.payload_version,
        )
        _validate_text(command.target.display, self._policy.max_query_chars, "generation target")
        reference = await _safe_call(
            lambda: self._publisher.publish(
                JobSubmission(
                    operation="generate",
                    request_id=command.context.request_id,
                    owner=principal,
                    idempotency_key=command.idempotency_key,
                    payload_version=command.payload_version,
                    reference_dataset_fingerprint=command.reference_dataset_fingerprint,
                    accepted_at=self._clock.now(),
                    payload={
                        "display": command.target.display,
                        "cambridge_id": command.target.cambridge_id,
                        "lexi_word_id": command.target.lexi_word_id,
                    },
                    maximum_age_seconds=int(self._policy.maximum_job_age.total_seconds()),
                    max_retries=self._policy.max_retries,
                )
            )
        )
        log_event(
            logger,
            "job_submitted",
            request_id=command.context.request_id,
            job_id=reference.job_id,
            operation="generate",
        )
        return reference

    async def submit_translation(self, command: SubmitTranslation) -> JobReference:
        principal = self._validate_submission(
            command.context.principal,
            command.context.deadline,
            command.idempotency_key,
            command.reference_dataset_fingerprint,
            command.payload_version,
        )
        _validate_text(command.source_kind, self._policy.max_query_chars, "source_kind")
        _validate_text(command.language, self._policy.max_query_chars, "language")
        _validate_text(command.source_hash, self._policy.max_query_chars, "source_hash")
        if command.source_id <= 0:
            raise public_error(ErrorCode.VALIDATION, "source_id is invalid.")
        reference = await _safe_call(
            lambda: self._publisher.publish(
                JobSubmission(
                    operation="translate",
                    request_id=command.context.request_id,
                    owner=principal,
                    idempotency_key=command.idempotency_key,
                    payload_version=command.payload_version,
                    reference_dataset_fingerprint=command.reference_dataset_fingerprint,
                    accepted_at=self._clock.now(),
                    payload={
                        "source_kind": command.source_kind,
                        "source_id": command.source_id,
                        "language": command.language,
                        "source_hash": command.source_hash,
                    },
                    maximum_age_seconds=int(self._policy.maximum_job_age.total_seconds()),
                    max_retries=self._policy.max_retries,
                )
            )
        )
        log_event(
            logger,
            "job_submitted",
            request_id=command.context.request_id,
            job_id=reference.job_id,
            operation="translate",
        )
        return reference

    def _validate_submission(
        self, principal, deadline, idempotency_key, fingerprint, payload_version
    ) -> Principal:
        owner = _require_principal(principal)
        _validate_text(idempotency_key, self._policy.max_idempotency_key_chars, "idempotency_key")
        if payload_version != 1:
            raise public_error(ErrorCode.PRECONDITION_FAILED, "Payload version is not supported.")
        if deadline is not None and deadline <= self._clock.now():
            raise public_error(ErrorCode.DEADLINE_EXCEEDED, "Request deadline elapsed.")
        if fingerprint != self._dataset.fingerprint:
            raise public_error(ErrorCode.PRECONDITION_FAILED, "Reference dataset does not match.")
        return owner


class ExecutionService:
    """Worker-only effects. It intentionally has no job-publisher dependency."""

    def __init__(
        self,
        lexicon: LexiconPort,
        dataset: ReferenceDataset,
        provider_gate: ProviderGate,
        policy: ServicePolicy,
        clock: Clock,
        source_preconditions: SourcePreconditionVerifier,
    ):
        self._lexicon = lexicon
        self._dataset = dataset
        self._provider_gate = provider_gate
        self._policy = policy
        self._clock = clock
        self._source_preconditions = source_preconditions

    async def execute_generate(self, command: ExecuteGenerate) -> Entry:
        self._validate_job(command.job, "generate", command.attempt)
        if (
            command.job.payload.get("display") != command.target.display
            or command.job.payload.get("cambridge_id") != command.target.cambridge_id
            or command.job.payload.get("lexi_word_id") != command.target.lexi_word_id
        ):
            raise public_error(ErrorCode.PRECONDITION_FAILED, "Job payload does not match.")
        generate = getattr(self._lexicon, "generate_fenced", self._lexicon.generate)
        return await self._run_provider(
            lambda: generate(command.target), command.job.owner.ownership_key
        )

    async def execute_translation(self, command: ExecuteTranslation) -> str:
        self._validate_job(command.job, "translate", command.attempt)
        expected = {
            "source_kind": command.source_kind,
            "source_id": command.source_id,
            "language": command.language,
            "source_hash": command.source_hash,
        }
        if any(command.job.payload.get(key) != value for key, value in expected.items()):
            raise public_error(ErrorCode.PRECONDITION_FAILED, "Job payload does not match.")
        matches = await _safe_call(
            lambda: self._source_preconditions.matches(
                command.source_kind, command.source_id, command.source_hash
            )
        )
        if not matches:
            raise public_error(ErrorCode.PRECONDITION_FAILED, "Source content changed.")
        return await self._run_provider(
            lambda: self._lexicon.translate_field(
                command.source_kind, command.source_id, command.language
            ),
            command.job.owner.ownership_key,
        )

    def _validate_job(self, job: JobSubmission, operation: str, attempt: int) -> None:
        if job.operation != operation or job.payload_version != 1:
            raise public_error(ErrorCode.PRECONDITION_FAILED, "Job contract is not supported.")
        if job.reference_dataset_fingerprint != self._dataset.fingerprint:
            raise public_error(ErrorCode.PRECONDITION_FAILED, "Reference dataset does not match.")
        if attempt < 1 or attempt > job.max_retries + 1:
            raise public_error(ErrorCode.PRECONDITION_FAILED, "Job retry cap exceeded.")
        if self._clock.now() >= job.accepted_at + timedelta(seconds=job.maximum_age_seconds):
            raise public_error(ErrorCode.DEADLINE_EXCEEDED, "Job maximum age elapsed.")

    async def _run_provider(self, operation, tenant: str):
        try:
            async with self._provider_gate.acquire(tenant):
                return await asyncio.wait_for(
                    operation(), timeout=self._policy.provider_attempt_timeout.total_seconds()
                )
        except asyncio.TimeoutError as exc:
            raise public_error(
                ErrorCode.DEADLINE_EXCEEDED, "Provider attempt timed out.", retryable=True
            ) from exc
        except ApplicationError:
            raise
        except Exception as exc:
            raise ApplicationError(to_public_error(exc)) from exc


class LocalProviderGate:
    """In-process fallback with a global and a per-tenant provider cap."""

    def __init__(self, maximum: int, maximum_per_tenant: int | None = None):
        self._semaphore = asyncio.Semaphore(maximum)
        self._maximum_per_tenant = maximum_per_tenant or maximum
        self._tenant_semaphores: dict[str, asyncio.Semaphore] = {}

    @asynccontextmanager
    async def acquire(self, tenant: str):
        tenant_semaphore = self._tenant_semaphores.setdefault(
            tenant, asyncio.Semaphore(self._maximum_per_tenant)
        )
        async with self._semaphore:
            async with tenant_semaphore:
                yield
