from lexi_service.transport.health import dependencies_ready


async def test_readiness_fails_when_any_dependency_is_unhealthy():
    health = dependencies_ready(lambda: _true(), lambda: _false(), lambda: _true())
    assert not await health.readiness()


async def _true():
    return True


async def _false():
    return False
