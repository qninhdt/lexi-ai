"""Word-owned PostgreSQL trigram indexes on the three lexical search sources."""

from sqlalchemy import text

TRIGRAM_SCHEMA = text(
    "SELECT n.nspname FROM pg_catalog.pg_extension e "
    "JOIN pg_catalog.pg_namespace n ON n.oid=e.extnamespace WHERE e.extname='pg_trgm'"
)
TRIGRAM_INDEXES = {
    table: f"ix_{table}_match_key_trigram" for table in ("words", "word_aliases", "sense_forms")
}


def managed_search_object(name, kind):
    return kind == "index" and name in TRIGRAM_INDEXES.values()


def install_search_index(connection):
    if connection.dialect.name != "postgresql":
        return  # SQLite is a lexical-only test backend, not a fuzzy search engine.
    connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public")
    schema = connection.execute(TRIGRAM_SCHEMA).scalar_one()
    quoted = connection.dialect.identifier_preparer.quote_schema(schema)
    for table, index in TRIGRAM_INDEXES.items():
        predicate = " WHERE generation_state='done'" if table == "words" else ""
        connection.exec_driver_sql(
            f"CREATE INDEX IF NOT EXISTS {index} ON {table} "
            f"USING gin (match_key {quoted}.gin_trgm_ops){predicate}"
        )


def remove_search_index(connection):
    if connection.dialect.name == "postgresql":
        for index in TRIGRAM_INDEXES.values():
            connection.exec_driver_sql(f"DROP INDEX IF EXISTS {index}")
        # pg_trgm is database-wide and may be used by other applications.
