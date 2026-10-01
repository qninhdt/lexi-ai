"""Deterministic exact-match scoring and explicit-price accounting."""

import math
from collections import defaultdict
from decimal import Decimal


def validate_pricing(pricing):
    if pricing is None:
        return
    if not isinstance(pricing, dict):
        raise ValueError("pricing must be an object or null")
    if not isinstance(pricing.get("currency"), str) or not pricing["currency"].strip():
        raise ValueError("pricing.currency is required")
    models = pricing.get("model_ids")
    if (
        not isinstance(models, list)
        or not models
        or any(not isinstance(model, str) or not model.strip() for model in models)
    ):
        raise ValueError("pricing.model_ids must list actual response model IDs")
    rates = (
        "input_per_million",
        "output_per_million",
        "cache_read_per_million",
        "cache_write_per_million",
        "per_request",
    )
    for name in rates:
        value = pricing.get(name)
        if value is not None and (
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
        ):
            raise ValueError(f"pricing.{name} must be a nonnegative finite number or null")
    if pricing.get("per_request") is not None and any(
        pricing.get(name) is not None for name in rates[:-1]
    ):
        raise ValueError("choose per_request OR token rates, not both")


def case_cost(record, pricing):
    """Return Decimal cost, or None when billing metadata/rates are insufficient."""
    if pricing is None or not record["usage"]:
        return None
    if any(item["model_id"] not in pricing["model_ids"] for item in record["usage"]):
        return None
    if pricing.get("per_request") is not None:
        # Benchmark clients do not retry. An unsuccessful request may or may not
        # be billed; preserve that uncertainty rather than inventing a charge.
        return Decimal(str(pricing["per_request"])) if record["status"] == "ok" else None
    input_rate, output_rate = pricing.get("input_per_million"), pricing.get("output_per_million")
    if input_rate is None or output_rate is None:
        return None
    input_rate, output_rate = Decimal(str(input_rate)), Decimal(str(output_rate))
    total = Decimal(0)
    for item in record["usage"]:
        input_tokens, output_tokens = item["input_tokens"], item["output_tokens"]
        if input_tokens is None or output_tokens is None:
            return None
        cost = input_tokens * input_rate + output_tokens * output_rate
        discounted_tokens = 0
        for count_name, rate_name in (
            ("cache_read_tokens", "cache_read_per_million"),
            ("cache_write_tokens", "cache_write_per_million"),
        ):
            rate = pricing.get(rate_name)
            rate = input_rate if rate is None else Decimal(str(rate))
            count = item[count_name]
            if count is not None:
                discounted_tokens += count
            if rate != input_rate:
                if count is None:
                    return None
                # Input already includes cached tokens; replace their rate,
                # never add their full charge on top of the input charge.
                cost += count * (rate - input_rate)
        if discounted_tokens > input_tokens:
            return None
        total += cost / 1_000_000
    return total


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(records, pricing=None):
    """Macro task exact match, complete totals or unknown, and all-case timing."""
    if not records:
        raise ValueError("no benchmark results")
    validate_pricing(pricing)
    tasks = defaultdict(lambda: {"cases": 0, "correct": 0, "errors": 0})
    fields = ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens")
    known = dict.fromkeys(fields, 0)
    complete = dict.fromkeys(fields, True)
    cost_total = Decimal(0)
    cost_known_cases = 0
    token_known_cases = 0
    model_ids = set()
    for record in records:
        task = tasks[record["task"]]
        task["cases"] += 1
        correct = record["status"] == "ok" and record["prediction"] == record["expected"]
        task["correct"] += correct
        task["errors"] += record["status"] != "ok"
        items = record["usage"]
        for field in fields:
            if not items or any(item[field] is None for item in items):
                complete[field] = False
            known[field] += sum(item[field] for item in items if item[field] is not None)
        if items and all(
            item["input_tokens"] is not None and item["output_tokens"] is not None for item in items
        ):
            token_known_cases += 1
        model_ids.update(item["model_id"] for item in items if item["model_id"] is not None)
        cost = case_cost(record, pricing)
        if cost is not None:
            cost_total += cost
            cost_known_cases += 1
    for task in tasks.values():
        task["accuracy"] = task["correct"] / task["cases"]
    tokens_complete = complete["input_tokens"] and complete["output_tokens"]
    latencies = [record["latency_ms"] for record in records]
    success_latencies = [record["latency_ms"] for record in records if record["status"] == "ok"]
    return {
        "accuracy": sum(task["accuracy"] for task in tasks.values()) / len(tasks),
        "accuracy_metric": "macro_task_exact_match",
        "cases": len(records),
        "errors": sum(task["errors"] for task in tasks.values()),
        "by_task": dict(sorted(tasks.items())),
        "actual_model_ids": sorted(model_ids),
        "total_tokens": known["input_tokens"] + known["output_tokens"] if tokens_complete else None,
        "token_counts": {field: known[field] if complete[field] else None for field in fields},
        "known_token_counts": known,
        "tokens_complete": tokens_complete,
        "token_known_cases": token_known_cases,
        "total_cost": float(cost_total) if cost_known_cases == len(records) else None,
        "known_cost": float(cost_total),
        "cost_complete": cost_known_cases == len(records),
        "cost_known_cases": cost_known_cases,
        "currency": pricing["currency"] if pricing else None,
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_mean_ms": sum(latencies) / len(latencies),
        "success_latency_p95_ms": percentile(success_latencies, 0.95)
        if success_latencies
        else None,
    }
