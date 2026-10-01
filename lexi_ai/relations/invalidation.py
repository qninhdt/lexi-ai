"""Relation-owned invalidation, including neutral evidence edits outside the ORM."""

_EVENTS = (
    ("definitions", "INSERT", "AFTER", "NEW.theme_id IS NULL", "NEW.sense_id"),
    (
        "definitions",
        "UPDATE",
        "AFTER",
        "(OLD.theme_id IS NULL OR NEW.theme_id IS NULL) AND "
        "(OLD.theme_id IS NOT NEW.theme_id OR OLD.sense_id IS NOT NEW.sense_id "
        "OR OLD.content IS NOT NEW.content)",
        "OLD.sense_id, NEW.sense_id",
    ),
    ("definitions", "DELETE", "AFTER", "OLD.theme_id IS NULL", "OLD.sense_id"),
    ("senses", "INSERT", "AFTER", "1", "NEW.id"),
    (
        "senses",
        "UPDATE",
        "AFTER",
        "OLD.word_id IS NOT NEW.word_id OR OLD.pos IS NOT NEW.pos",
        "OLD.id, NEW.id",
    ),
    ("senses", "DELETE", "BEFORE", "1", "OLD.id"),
)


def _affected_rows(table, operation):
    """Transition rows whose neutral evidence or Sense identity actually changed."""
    columns = ("theme_id", "sense_id", "content") if table == "definitions" else ("word_id", "pos")
    versions = (
        ("new",) if operation == "INSERT" else ("old",) if operation == "DELETE" else ("old", "new")
    )
    branches = []
    for version in versions:
        alias = f"{version}_row"
        conditions = [f"{alias}.theme_id IS NULL"] if table == "definitions" else []
        source = f"{version}_rows AS {alias}"
        if operation == "UPDATE":
            other = "new" if version == "old" else "old"
            other_alias = f"{other}_row"
            source += f" LEFT JOIN {other}_rows AS {other_alias} ON {alias}.id={other_alias}.id"
            left = ",".join(f"{alias}.{column}" for column in columns)
            right = ",".join(f"{other_alias}.{column}" for column in columns)
            conditions.append(
                f"({other_alias}.id IS NULL OR ROW({left}) IS DISTINCT FROM ROW({right}))"
            )
        fields = (
            f"{alias}.sense_id AS id" if table == "definitions" else f"{alias}.id, {alias}.word_id"
        )
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        branches.append(f"SELECT {fields} FROM {source}{where}")
    return " UNION ".join(branches)


def _install_postgres(connection, table, operation):
    name = _trigger_name(table, operation)
    affected = _affected_rows(table, operation)
    word_column = "affected.word_id" if table == "senses" else "senses.word_id"
    owners = "" if table == "senses" else " LEFT JOIN senses ON senses.id=affected.id"
    connection.exec_driver_sql(f"""
        CREATE OR REPLACE FUNCTION {name}() RETURNS trigger
        LANGUAGE plpgsql SET search_path FROM CURRENT AS $$
        DECLARE sense_ids integer[]; word_ids integer[];
        BEGIN
            WITH affected AS ({affected})
            SELECT array_agg(DISTINCT affected.id), array_agg(DISTINCT {word_column})
              INTO sense_ids, word_ids FROM affected{owners};
            IF sense_ids IS NULL THEN RETURN NULL; END IF;
            -- Include pending edges: these locks serialize concurrent resolver applies.
            UPDATE sense_relations
               SET to_sense_id=NULL, target_hash=NULL, resolve_attempted_at=NULL
             WHERE from_sense_id = ANY(sense_ids) OR to_word_id = ANY(word_ids);
            RETURN NULL;
        END; $$
    """)
    references = {
        "INSERT": "NEW TABLE AS new_rows",
        "UPDATE": "OLD TABLE AS old_rows NEW TABLE AS new_rows",
        "DELETE": "OLD TABLE AS old_rows",
    }[operation]
    connection.exec_driver_sql(f"DROP TRIGGER IF EXISTS {name} ON {table}")
    connection.exec_driver_sql(
        f"CREATE TRIGGER {name} AFTER {operation} ON {table} "
        f"REFERENCING {references} FOR EACH STATEMENT EXECUTE FUNCTION {name}()"
    )


def _trigger_name(table, operation):
    return f"lexi_relation_{table}_{operation.lower()}"


def install_relation_triggers(connection):
    """Invalidate once per affected set on PostgreSQL; SQLite serializes row writes."""
    postgres = connection.dialect.name == "postgresql"
    for table, operation, timing, condition, sense_ids in _EVENTS:
        name = _trigger_name(table, operation)
        if postgres:
            _install_postgres(connection, table, operation)
        else:
            word_ids = f"SELECT word_id FROM senses WHERE id IN ({sense_ids})"
            if table == "senses":
                word_ids = {
                    "INSERT": "NEW.word_id",
                    "UPDATE": "OLD.word_id, NEW.word_id",
                    "DELETE": "OLD.word_id",
                }[operation]
            connection.exec_driver_sql(f"""
                CREATE TRIGGER IF NOT EXISTS {name} {timing} {operation} ON {table}
                WHEN {condition}
                BEGIN
                    UPDATE sense_relations
                       SET to_sense_id = NULL, target_hash = NULL, resolve_attempted_at = NULL
                     WHERE (resolve_attempted_at IS NOT NULL OR to_sense_id IS NOT NULL)
                       AND (from_sense_id IN ({sense_ids}) OR to_word_id IN ({word_ids}));
                END
            """)


def remove_relation_triggers(connection):
    for table, operation, *_rest in _EVENTS:
        name = _trigger_name(table, operation)
        suffix = f" ON {table}" if connection.dialect.name == "postgresql" else ""
        connection.exec_driver_sql(f"DROP TRIGGER IF EXISTS {name}{suffix}")
        if connection.dialect.name == "postgresql":
            connection.exec_driver_sql(f"DROP FUNCTION IF EXISTS {name}()")
