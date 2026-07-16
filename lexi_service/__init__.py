"""Transport-neutral service boundary for Lexi-AI.

Adapters for gRPC and HTTP belong in later phases. Importing this package does
not construct a library engine or import transport, broker, or worker clients.
"""

from lexi_service.application.services import ExecutionService, QueryService, SubmissionService
from lexi_service.runtime import ServiceRuntime

__all__ = ["ExecutionService", "QueryService", "ServiceRuntime", "SubmissionService"]
