"""LanceDB-backed vector index: embedded, on disk, no server to run.

LanceDB is a synchronous, file-based store, so every call here is wrapped in
:func:`asyncio.to_thread` — the blocking detail stays inside the adapter and the
port stays async. ``lancedb`` and ``pyarrow`` are imported lazily to keep module
import cheap; the extra's presence is checked up front by ``build_vector_index``,
so reaching this module means the dependency is there.

The table is created on the first write, with a fixed-width vector column: the
dimension is a schema fact in Lance, so it comes from the first batch and a later
batch of a different width is rejected rather than silently corrupting the index.
"""

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path

from lexi_ai.domain.models import VectorHit, VectorRecord
from lexi_ai.infrastructure.vectors.validation import uniform_dimension

# One table holds every sense vector. The encoder's model name is a column, not a
# table, so switching models is a filter change rather than a migration.
TABLE = "sense_vectors"
_ID = "id"
_VECTOR = "vector"


class LanceDbVectorIndex:
    """The default durable index. One table, one row per sense, id-keyed upserts."""

    def __init__(self, path: str, metric: str = "cosine") -> None:
        self._path = path
        self._metric = metric
        self._table = None

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        if not records:
            return 0
        dim = uniform_dimension(records)
        rows = [
            {_ID: record.id, _VECTOR: [float(value) for value in record.vector], **record.meta}
            for record in records
        ]
        return await asyncio.to_thread(self._upsert_sync, rows, dim)

    async def query(
        self, vector: Sequence[float], k: int, where: Mapping[str, str] | None = None
    ) -> list[VectorHit]:
        if k <= 0:
            return []
        return await asyncio.to_thread(
            self._query_sync, [float(value) for value in vector], k, where
        )

    async def delete(self, ids: Sequence[str]) -> int:
        if not ids:
            return 0
        return await asyncio.to_thread(self._delete_sync, list(ids))

    async def ids(self, where: Mapping[str, str] | None = None) -> set[str]:
        return await asyncio.to_thread(self._ids_sync, where)

    async def fetch(self, ids: Sequence[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        return await asyncio.to_thread(self._fetch_sync, list(ids))

    # --- synchronous LanceDB access (always via to_thread) ----------------

    def _upsert_sync(self, rows: list[dict], dim: int) -> int:
        table = self._open(create_with_dim=dim, meta_keys=self._meta_keys(rows))
        (
            table.merge_insert(_ID)
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )
        return len(rows)

    def _query_sync(self, vector: list[float], k: int, where: Mapping[str, str] | None):  # noqa: ANN202
        table = self._open()
        if table is None:
            return []
        search = table.search(vector).metric(self._metric).limit(k)
        clause = _where_clause(where)
        if clause:
            # Pre-filter so the ANN scan itself is restricted; post-filtering would
            # let a wrong-model vector consume one of the k slots.
            search = search.where(clause, prefilter=True)
        # Lance reports cosine DISTANCE; the port's contract is similarity.
        return [
            VectorHit(id=row[_ID], score=1.0 - float(row["_distance"])) for row in search.to_list()
        ]

    def _delete_sync(self, ids: list[str]) -> int:
        table = self._open()
        if table is None:
            return 0
        quoted = ", ".join(f"'{_escape(stored_id)}'" for stored_id in ids)
        result = table.delete(f"{_ID} IN ({quoted})")
        return int(getattr(result, "num_deleted_rows", 0) or 0)

    def _ids_sync(self, where: Mapping[str, str] | None) -> set[str]:
        table = self._open()
        if table is None:
            return set()
        # A projected full scan: O(N) on one string column, and it runs during
        # backfill (maintenance), never on the query path.
        scan = table.search().select([_ID]).limit(0)
        clause = _where_clause(where)
        if clause:
            scan = scan.where(clause)
        return {row[_ID] for row in scan.to_list()}

    def _fetch_sync(self, ids: list[str]) -> dict[str, list[float]]:
        table = self._open()
        if table is None:
            return {}
        quoted = ", ".join(f"'{_escape(stored_id)}'" for stored_id in ids)
        rows = (
            table.search().where(f"{_ID} IN ({quoted})").select([_ID, _VECTOR]).limit(0).to_list()
        )
        return {row[_ID]: [float(value) for value in row[_VECTOR]] for row in rows}

    def _open(self, create_with_dim: int | None = None, meta_keys: Sequence[str] = ()):  # noqa: ANN202
        """The table, opened once. ``None`` when it does not exist yet and no
        dimension was supplied to create it (a read before the first write)."""
        if self._table is not None:
            return self._table
        import lancedb

        Path(self._path).mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(self._path)
        if TABLE in _existing_tables(db):
            self._table = db.open_table(TABLE)
        elif create_with_dim is not None:
            self._table = db.create_table(TABLE, schema=_schema(create_with_dim, meta_keys))
        return self._table

    @staticmethod
    def _meta_keys(rows: list[dict]) -> list[str]:
        return sorted({key for row in rows for key in row if key not in (_ID, _VECTOR)})


def _existing_tables(db) -> set[str]:  # noqa: ANN001 - a lancedb connection
    """The table names in a connection, across the two shapes lancedb has used.

    ``table_names`` returned a plain list; ``list_tables`` returns a paginated
    response object. Normalizing here keeps the extra's version range wide.
    """
    lister = getattr(db, "list_tables", None) or db.table_names
    result = lister()
    tables = getattr(result, "tables", result)
    return {getattr(table, "name", table) for table in tables}


def _schema(dim: int, meta_keys: Sequence[str]):  # noqa: ANN202
    """A fixed-width vector column plus one string column per metadata key."""
    import pyarrow as pa

    fields = [
        pa.field(_ID, pa.string()),
        pa.field(_VECTOR, pa.list_(pa.float32(), dim)),
    ]
    fields += [pa.field(key, pa.string()) for key in meta_keys]
    return pa.schema(fields)


def _where_clause(where: Mapping[str, str] | None) -> str:
    """Render an equality filter as the SQL fragment LanceDB expects."""
    if not where:
        return ""
    return " AND ".join(f"{key} = '{_escape(value)}'" for key, value in sorted(where.items()))


def _escape(value: str) -> str:
    """Escape a single-quoted SQL literal. Metadata is internal, but the filter is
    still built as text, so the quote doubling is not optional."""
    return value.replace("'", "''")
