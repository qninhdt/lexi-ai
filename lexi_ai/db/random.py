"""Uniform random lookup over maintained dense slots, entirely inside one SELECT."""

from sqlalchemy import Integer, cast, func, select
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import FunctionElement


class RandomFraction(FunctionElement):
    inherit_cache = True


@compiles(RandomFraction, "postgresql")
def _random_pg(element, compiler, **kwargs):
    return "random()"


@compiles(RandomFraction, "sqlite")
def _random_sqlite(element, compiler, **kwargs):
    # SQLite random() is an integer, not PostgreSQL's fraction in [0, 1).
    return "((random() & 9007199254740991) / 9007199254740992.0)"


def last_position(table, slot, *scope, scope_order=(), correlate=()):
    # Include nullable scope keys in the ordering. IS NULL alone does not let
    # PostgreSQL infer that an index ordered by (scope, slot) orders by slot alone.
    return (
        select(slot)
        .select_from(table)
        .where(*scope)
        .order_by(*(column.desc() for column in scope_order), slot.desc())
        .limit(1)
        .correlate(*correlate)
        .scalar_subquery()
    )


def random_position(table, slot, *scope, scope_order=(), correlate=()):
    last = last_position(table, slot, *scope, scope_order=scope_order, correlate=correlate)
    # FLOOR before INTEGER cast: PostgreSQL float->integer rounds rather than truncates.
    return (
        select(cast(func.floor(RandomFraction() * func.coalesce(last, 0)), Integer) + 1)
        .correlate(*correlate)
        .scalar_subquery()
    )
