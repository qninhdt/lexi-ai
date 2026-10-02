"""Exact Theme key normalization and SQL namespace selection."""

from sqlalchemy import literal, select

from lexi_ai.errors import InvalidResourceError
from lexi_ai.schema import Theme


def normalize_key(value: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 255:
        raise InvalidResourceError("invalid Theme key")
    return " ".join(value.casefold().split())


def theme_scope(theme_id=None, theme_key=None):
    if theme_key is not None:
        identifier = select(Theme.id).where(Theme.key == normalize_key(theme_key)).scalar_subquery()
    else:
        identifier = theme_id
    valid = (
        literal(True)
        if theme_id is None and theme_key is None
        else select(Theme.id).where(Theme.id == identifier).exists()
    )
    return identifier, valid
