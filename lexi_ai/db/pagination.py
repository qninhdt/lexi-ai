"""Validation shared by exclusive-ID keyset readers."""

from lexi_ai.errors import InvalidResourceError


def validate_page(limit=None, after_id=None):
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 500):
        raise InvalidResourceError("page limit must be an integer in 1..500")
    if after_id is not None and type(after_id) is not int:
        raise InvalidResourceError("invalid page cursor")
