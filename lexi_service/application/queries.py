"""Transport-neutral synchronous query DTOs."""

from dataclasses import dataclass

from lexi_service.application.commands import RequestContext


@dataclass(frozen=True)
class SearchQuery:
    context: RequestContext
    query: str


@dataclass(frozen=True)
class LookupEntryQuery:
    context: RequestContext
    word_id: int


@dataclass(frozen=True)
class GetJobQuery:
    context: RequestContext
    job_id: str


@dataclass(frozen=True)
class StatsQuery:
    context: RequestContext
