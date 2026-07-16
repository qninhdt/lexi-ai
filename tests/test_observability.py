from lexi_service.observability.logging import safe_fields
from lexi_service.observability.metrics import ServiceMetrics


def test_sensitive_fields_are_not_available_to_logs():
    assert safe_fields(request_id="r1", prompt="secret", api_key="key") == {
        "request_id": "r1",
        "prompt": "[redacted]",
        "api_key": "[redacted]",
    }


def test_metrics_are_named_counters():
    metrics = ServiceMetrics()
    metrics.increment("jobs.claimed")
    metrics.increment("jobs.claimed")
    assert metrics.snapshot() == {"jobs.claimed": 2}


def test_metrics_render_without_user_controlled_labels():
    metrics = ServiceMetrics()
    metrics.increment("lexi_http_requests_total")
    metrics.add("lexi_http_status_200_total", 2)
    assert metrics.prometheus() == (
        "# TYPE lexi_http_requests_total counter\nlexi_http_requests_total 1\n"
        "# TYPE lexi_http_status_200_total counter\nlexi_http_status_200_total 2\n"
    )
