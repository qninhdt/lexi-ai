"""Defaults for bounded provider and lexical operations."""

import re


def database_schema_name(value: str | None) -> str | None:
    if value is not None and (
        not isinstance(value, str)
        or len(value) > 63
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)
    ):
        raise ValueError("invalid generated DB schema name")
    return value


MAX_QUERY_LENGTH = 256
MAX_LEMMA_LENGTH = 256
MAX_TEXT_LENGTH = 16_000
MAX_PROMPT_LENGTH = 64_000
MAX_PATTERN_LENGTH = 256
