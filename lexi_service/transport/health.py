"""Liveness is public; readiness is authenticated and dependency-aware."""

from collections.abc import Awaitable, Callable


class HealthChecks:
    def __init__(self, ready: Callable[[], Awaitable[bool]]):
        self._ready = ready

    async def readiness(self) -> bool:
        return await self._ready()


def dependencies_ready(*checks: Callable[[], Awaitable[bool]]) -> HealthChecks:
    async def ready() -> bool:
        for check in checks:
            if not await check():
                return False
        return True

    return HealthChecks(ready)
