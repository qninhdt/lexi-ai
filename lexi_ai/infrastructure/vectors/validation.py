"""Checks every vector adapter owes its callers, regardless of backend."""

from collections.abc import Sequence

from lexi_ai.domain.models import VectorRecord


def uniform_dimension(records: Sequence[VectorRecord]) -> int:
    """The shared dimension of a batch, raising when the batch is not uniform.

    A mixed-width batch is a caller bug (two encoders in one call), and it must fail
    loudly: in a fixed-width store it would be a schema error at some later write,
    and in a scan-based one it would silently score as zero forever.
    """
    dim = len(records[0].vector)
    if any(len(record.vector) != dim for record in records):
        raise ValueError("every vector in one upsert must have the same dimension")
    return dim
