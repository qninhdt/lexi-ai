"""Structured logs that retain correlation while redacting sensitive values."""

import json
import logging

SENSITIVE_KEYS = frozenset({"authorization", "api_key", "prompt", "content", "payload", "token"})


def safe_fields(**fields: object) -> dict[str, object]:
    return {
        key: "[redacted]" if key.lower() in SENSITIVE_KEYS else value
        for key, value in fields.items()
    }


def log_event(logger: logging.Logger, event: str, **fields: object) -> None:
    logger.info(json.dumps({"event": event, **safe_fields(**fields)}, default=str, sort_keys=True))
