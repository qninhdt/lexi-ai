"""Ports owned by the service boundary, never by the `lexi_ai` library."""

from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from lexi_ai.read_models import Entry, SearchResult, Stats
from lexi_service.application.commands import JobReference, JobSubmission
from lexi_service.identity import Principal


@dataclass(frozen=True)
class JobRecord:
    reference: JobReference
    owner: Principal
    operation: str
    private_result: object | None = None


class LexiconPort(Protocol):
    async def search(self, query: str) -> list[SearchResult]: ...

    async def get_entry(self, word_id: int) -> Entry: ...

    async def generate(self, target: SearchResult | str) -> Entry: ...

    async def generate_fenced(self, target: SearchResult | str) -> Entry: ...

    async def source_hash(self, source_kind: str, source_id: int) -> str | None: ...

    async def translate_field(self, source_kind: str, source_id: int, lang: str) -> str: ...

    async def stats(self) -> Stats: ...


class JobPublisher(Protocol):
    async def publish(self, submission: JobSubmission) -> JobReference: ...


class JobReader(Protocol):
    async def get(self, job_id: str) -> JobRecord | None: ...


class ReferenceDataset(Protocol):
    @property
    def fingerprint(self) -> str: ...


class Clock(Protocol):
    def now(self) -> datetime: ...


class ProviderGate(Protocol):
    def acquire(self, tenant: str) -> AbstractAsyncContextManager[None]: ...


class SourcePreconditionVerifier(Protocol):
    async def matches(self, source_kind: str, source_id: int, expected_hash: str) -> bool: ...


class ResourceCloser(Protocol):
    async def close(self) -> None: ...
