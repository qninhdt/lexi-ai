"""In-process counters/timers with an OpenTelemetry-compatible naming surface."""

from collections import Counter


class ServiceMetrics:
    def __init__(self):
        self.counters: Counter[str] = Counter()

    def increment(self, name: str) -> None:
        self.counters[name] += 1

    def add(self, name: str, value: int) -> None:
        self.counters[name] += value

    def snapshot(self) -> dict[str, int]:
        return dict(self.counters)

    def prometheus(self) -> str:
        """Render integer process counters without user-controlled labels."""
        lines = []
        for name, value in sorted(self.counters.items()):
            metric = name.replace("-", "_")
            lines.append(f"# TYPE {metric} counter\n{metric} {value}")
        return "\n".join(lines) + ("\n" if lines else "")
