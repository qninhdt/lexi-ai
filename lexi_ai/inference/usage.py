"""Request-local accounting; no global state, pricing, persistence or domain rules."""

from collections.abc import Mapping
from dataclasses import fields

from ..models import TokenUsage


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _count(value):
    # Malformed accounting must never look like a valid charge or a zero-token call.
    return value if type(value) is int and value >= 0 else None


def _model(response):
    value = _get(response, "model")
    return value if isinstance(value, str) and value.strip() else None


def openai_usage(response) -> TokenUsage:
    usage = _get(response, "usage")
    details = _get(usage, "prompt_tokens_details", _get(usage, "input_tokens_details"))
    return TokenUsage(
        model_id=_model(response),
        input_tokens=_count(_get(usage, "prompt_tokens", _get(usage, "input_tokens"))),
        cache_read_tokens=_count(_get(details, "cached_tokens")),
        cache_write_tokens=_count(_get(usage, "cache_write_tokens")),
        output_tokens=_count(_get(usage, "completion_tokens", _get(usage, "output_tokens"))),
    )


def decision_usage(response) -> list[TokenUsage]:
    # The official adapter preserves raw provider responses, including unsuccessful
    # attempts, in its debug trace. Count attempts OR cumulative usage, never both.
    attempts = _get(_get(response, "debug"), "llm_attempts")
    if attempts:
        records = []
        for attempt in attempts:
            raw = _get(attempt, "llm_response")
            if _get(raw, "usage") is not None:
                records.append(openai_usage(raw))
            else:
                records.append(
                    TokenUsage(
                        # Adapter model_name can be a requested alias. Only a raw
                        # provider response can identify the actual fallback model.
                        model_id=_model(raw),
                        input_tokens=_count(_get(raw, "input_tokens")),
                        cache_read_tokens=_count(_get(raw, "cache_read_tokens")),
                        cache_write_tokens=_count(_get(raw, "cache_write_tokens")),
                        output_tokens=_count(_get(raw, "output_tokens")),
                    )
                )
        return records
    usage = _get(response, "usage")
    return [
        TokenUsage(
            model_id=_model(response),
            input_tokens=_count(_get(usage, "input_tokens_total", _get(usage, "input_tokens"))),
            cache_read_tokens=_count(_get(usage, "cache_read_tokens")),
            cache_write_tokens=_count(_get(usage, "cache_write_tokens")),
            output_tokens=_count(_get(usage, "output_tokens_total", _get(usage, "output_tokens"))),
        )
    ]


def merge_usage(records: list[TokenUsage]) -> list[TokenUsage]:
    """Sum known counts by actual model; an unreported count makes that sum unknown."""
    by_model = {}
    count_fields = [field.name for field in fields(TokenUsage) if field.name != "model_id"]
    for record in records:
        previous = by_model.get(record.model_id)
        if previous is None:
            by_model[record.model_id] = record
        else:
            counts = {}
            for name in count_fields:
                left, right = getattr(previous, name), getattr(record, name)
                counts[name] = None if left is None or right is None else left + right
            by_model[record.model_id] = TokenUsage(model_id=record.model_id, **counts)
    # Parallel relations can complete in any order; presentation stays deterministic.
    return [by_model[key] for key in sorted(by_model, key=lambda key: (key is None, key or ""))]


class UsageRecorder:
    """One local recorder per API call, shared only by that call's child requests."""

    def __init__(self, enabled: bool):
        if type(enabled) is not bool:
            raise TypeError("with_usage must be a boolean")
        self.enabled = enabled
        self.records: list[TokenUsage] = []

    def __enter__(self):
        return self

    def __exit__(self, _type, error, _traceback):
        if error is not None and self.enabled:
            error.usage = merge_usage(self.records)

    def wrap(self, provider):
        return _TrackedProvider(provider, self.records) if self.enabled else provider

    def finish[T](self, value: T) -> T | tuple[T, list[TokenUsage]]:
        return (value, merge_usage(self.records)) if self.enabled else value


class _TrackedProvider:
    """Opt-in provider protocol adapter; domain tasks keep their normal return values."""

    def __init__(self, provider, records):
        self.provider, self.records = provider, records

    async def _request(self, method, *args, **kwargs):
        try:
            value, usage = await method(*args, **kwargs, with_usage=True)
        except Exception as error:
            self.records.extend(getattr(error, "usage", None) or [])
            raise
        self.records.extend(usage)
        return value

    async def complete(self, *args, **kwargs):
        return await self._request(self.provider.complete, *args, **kwargs)

    async def decide(self, *args, **kwargs):
        return await self._request(self.provider.decide, *args, **kwargs)
