"""Question-owned dense slots: indexed max and random lookup, no COUNT/scan.

Question slots cover both all-type and type-filtered banks. Deletion moves only
the last slot into the gap; public IDs and public ID ordering never change.
Postgres writers lock the owning Sense; SQLite already serializes its writers.
Triggers also maintain slots for direct SQL inserts, deletes and scope moves.
"""

_SCOPES = {
    "questions": {
        "position": ("sense_id", "theme_id"),
        "type_position": ("sense_id", "theme_id", "question_type"),
    },
}


def _equal(columns, record, postgres, *, neutral=False):
    # Separate NULL Theme scopes so PostgreSQL can seek AND preserve slot ordering.
    return " AND ".join(
        "theme_id IS NULL"
        if postgres and column == "theme_id" and neutral
        else f"{column} {'=' if postgres else 'IS'} {record}.{column}"
        for column in columns
    )


def _theme_branch(record, neutral, themed):
    return f"IF {record}.theme_id IS NULL THEN {neutral} ELSE {themed} END IF;"


def install_positions(connection):
    postgres = connection.dialect.name == "postgresql"
    for table, scopes in _SCOPES.items():
        all_columns = tuple(dict.fromkeys(c for scope in scopes.values() for c in scope))
        changed = " OR ".join(
            f"OLD.{c} IS DISTINCT FROM NEW.{c}" if postgres else f"OLD.{c} IS NOT NEW.{c}"
            for c in all_columns
        )
        assign, repair = [], []
        for slot, columns in scopes.items():
            # Full scope order reuses the composite index for neutral banks too.
            ordering = ", ".join(f"{column} DESC" for column in (*columns, slot))
            assignments, repairs = [], []
            for neutral in (True, False) if postgres else (False,):
                maximum = (
                    f"SELECT {slot} FROM {table} "
                    f"WHERE {_equal(columns, 'NEW', postgres, neutral=neutral)} "
                    f"AND id <> NEW.id ORDER BY {ordering} LIMIT 1"
                )
                assignments.append(
                    f"NEW.{slot} := COALESCE(({maximum}), 0) + 1;"
                    if postgres
                    else f"{slot} = COALESCE(({maximum}), 0) + 1"
                )
                last = (
                    f"SELECT id FROM {table} "
                    f"WHERE {_equal(columns, 'OLD', postgres, neutral=neutral)} "
                    f"AND {slot} > OLD.{slot} ORDER BY {ordering} LIMIT 1"
                )
                repairs.append(f"UPDATE {table} SET {slot} = OLD.{slot} WHERE id = ({last});")
            assign.append(_theme_branch("NEW", *assignments) if postgres else assignments[0])
            repair.append(_theme_branch("OLD", *repairs) if postgres else repairs[0])
        name = f"lexi_positions_{table}"
        if postgres:
            connection.exec_driver_sql(f"""
                CREATE OR REPLACE FUNCTION {name}() RETURNS trigger
                LANGUAGE plpgsql SET search_path FROM CURRENT AS $$
                BEGIN
                  IF TG_WHEN = 'BEFORE' THEN
                    IF TG_OP = 'INSERT' THEN
                      PERFORM id FROM senses WHERE id=NEW.sense_id FOR UPDATE;
                      {" ".join(assign)}
                    ELSIF TG_OP = 'DELETE' THEN
                      PERFORM id FROM senses WHERE id=OLD.sense_id FOR UPDATE;
                      RETURN OLD;
                    ELSIF {changed} THEN
                      PERFORM id FROM senses WHERE id IN (OLD.sense_id, NEW.sense_id)
                        ORDER BY id FOR UPDATE;
                      {" ".join(assign)}
                    END IF;
                    RETURN NEW;
                  END IF;
                  IF TG_OP = 'DELETE' THEN
                    {" ".join(repair)}
                    RETURN OLD;
                  ELSIF {changed} THEN
                    {" ".join(repair)}
                  END IF;
                  RETURN NEW;
                END; $$
            """)
            for timing, events in (
                ("BEFORE", "INSERT OR UPDATE OR DELETE"),
                ("AFTER", "UPDATE OR DELETE"),
            ):
                connection.exec_driver_sql(
                    f"DROP TRIGGER IF EXISTS {name}_{timing.lower()} ON {table}"
                )
                connection.exec_driver_sql(
                    f"CREATE TRIGGER {name}_{timing.lower()} {timing} {events} ON {table} "
                    f"FOR EACH ROW EXECUTE FUNCTION {name}()"
                )
        else:
            for suffix in ("insert", "update", "delete"):
                connection.exec_driver_sql(f"DROP TRIGGER IF EXISTS {name}_{suffix}")
            connection.exec_driver_sql(f"""
                CREATE TRIGGER {name}_insert AFTER INSERT ON {table}
                BEGIN UPDATE {table} SET {", ".join(assign)} WHERE id=NEW.id; END
            """)
            connection.exec_driver_sql(f"""
                CREATE TRIGGER {name}_update AFTER UPDATE ON {table} WHEN {changed}
                BEGIN
                  UPDATE {table} SET {", ".join(assign)} WHERE id=NEW.id;
                  {" ".join(repair)}
                END
            """)
            connection.exec_driver_sql(f"""
                CREATE TRIGGER {name}_delete AFTER DELETE ON {table}
                BEGIN {" ".join(repair)} END
            """)


def remove_positions(connection):
    postgres = connection.dialect.name == "postgresql"
    for table in _SCOPES:
        name = f"lexi_positions_{table}"
        for suffix in ("before", "after") if postgres else ("insert", "update", "delete"):
            connection.exec_driver_sql(
                f"DROP TRIGGER IF EXISTS {name}_{suffix}" + (f" ON {table}" if postgres else "")
            )
        if postgres:
            connection.exec_driver_sql(f"DROP FUNCTION IF EXISTS {name}()")
