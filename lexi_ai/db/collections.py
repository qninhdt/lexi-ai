"""Ordered, fresh child collections in one SQL statement; no stored JSON mirror."""

from sqlalchemy import JSON, literal, select
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import FunctionElement


class JsonObject(FunctionElement):
    type = JSON()
    inherit_cache = True


class JsonArray(FunctionElement):
    type = JSON()
    inherit_cache = True


@compiles(JsonObject, "postgresql")
def _object_pg(element, compiler, **kwargs):
    return f"jsonb_build_object({compiler.process(element.clauses, **kwargs)})"


@compiles(JsonObject, "sqlite")
def _object_sqlite(element, compiler, **kwargs):
    clauses = []
    for number, clause in enumerate(element.clauses):
        value = compiler.process(clause, **kwargs)
        if number % 2 and isinstance(clause.type, JSON):
            value = f"json({value})"
        clauses.append(value)
    return f"json_object({', '.join(clauses)})"


@compiles(JsonArray, "postgresql")
def _array_pg(element, compiler, **kwargs):
    return f"COALESCE(jsonb_agg({compiler.process(element.clauses, **kwargs)}), '[]'::jsonb)"


@compiles(JsonArray, "sqlite")
def _array_sqlite(element, compiler, **kwargs):
    return f"json_group_array(json({compiler.process(element.clauses, **kwargs)}))"


def collection(source, fields, *conditions, order_by, correlate=(), limit=None):
    """Aggregate an independently ordered child query, never Cartesian-join siblings."""
    item = JsonObject(*(part for key, value in fields.items() for part in (literal(key), value)))
    ordered = (
        select(item.label("item"))
        .select_from(source)
        .where(*conditions)
        .order_by(*order_by)
        .correlate(*correlate)
    )
    if limit is not None:
        ordered = ordered.limit(limit)
    ordered = ordered.subquery()
    return select(JsonArray(ordered.c.item)).scalar_subquery()
