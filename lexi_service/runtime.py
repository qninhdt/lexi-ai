"""Process-scoped runtime graph with explicit, idempotent async shutdown."""

import asyncio

from lexi_service.application.policies import ServicePolicy
from lexi_service.application.services import (
    ExecutionService,
    LocalProviderGate,
    QueryService,
    SubmissionService,
)
from lexi_service.ports import (
    Clock,
    JobPublisher,
    JobReader,
    LexiconPort,
    ProviderGate,
    ReferenceDataset,
    ResourceCloser,
    SourcePreconditionVerifier,
)


class ServiceRuntime:
    def __init__(
        self,
        queries: QueryService,
        submissions: SubmissionService,
        executions: ExecutionService,
        resources: ResourceCloser,
        policy: ServicePolicy | None = None,
        dataset: ReferenceDataset | None = None,
    ):
        self.queries = queries
        self.submissions = submissions
        self.executions = executions
        self._resources = resources
        self.policy = policy
        self.dataset = dataset
        self._closed = False
        self._close_lock = asyncio.Lock()

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            await self._resources.close()
            self._closed = True

    @classmethod
    def compose(
        cls,
        *,
        lexicon: LexiconPort,
        jobs: JobReader,
        publisher: JobPublisher,
        dataset: ReferenceDataset,
        policy: ServicePolicy,
        clock: Clock,
        resources: ResourceCloser,
        source_preconditions: SourcePreconditionVerifier,
        provider_gate: ProviderGate | None = None,
    ) -> "ServiceRuntime":
        """Build one reusable process graph around one `Lexicon` instance."""
        gate = provider_gate or LocalProviderGate(
            policy.max_provider_concurrency, policy.max_provider_concurrency_per_tenant
        )
        return cls(
            QueryService(lexicon, jobs, policy),
            SubmissionService(publisher, dataset, policy, clock),
            ExecutionService(lexicon, dataset, gate, policy, clock, source_preconditions),
            resources,
            policy,
            dataset,
        )

    async def __aenter__(self) -> "ServiceRuntime":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()
