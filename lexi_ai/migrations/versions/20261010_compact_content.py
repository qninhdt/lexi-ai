"""Convert stored target markup and remove unused relation glosses."""

import json
import re

import sqlalchemy as sa
from alembic import op

revision = "20261010_compact"
down_revision = "20261010_ref_forms"
branch_labels = None
depends_on = None

_CODES = {
    "base": "",
    "past": "p",
    "past_participle": "pp",
    "present_3sg": "3",
    "ing": "ing",
    "plural": "pl",
    "comparative": "c",
    "superlative": "s",
}
_OLD = re.compile(r'<t inf="([a-z0-9_]+)">([^<>]+)</t>', re.I)
_NEW = re.compile(r"\[([^\[\]|]+)(?:\|([a-z0-9]+))?\]", re.I)


def _convert(value, reverse=False):
    if isinstance(value, str):
        if reverse:
            labels = {code: label for label, code in _CODES.items()}
            return _NEW.sub(lambda m: f'<t inf="{labels[(m[2] or "").lower()]}">{m[1]}</t>', value)

        def tag(match):
            code = _CODES[match[1].lower()]
            return f"[{match[2]}{'|' + code if code else ''}]"

        return _OLD.sub(tag, value)
    if isinstance(value, list):
        return [_convert(item, reverse) for item in value]
    if isinstance(value, dict):
        return {key: _convert(item, reverse) for key, item in value.items()}
    return value


def _rewrite(reverse=False):
    connection = op.get_bind()
    for name, column in [
        ("examples", "content"),
        ("definitions", "content"),
        ("questions", "payload"),
    ]:
        table = sa.table(name, sa.column("id", sa.Integer), sa.column(column, sa.Text))
        for id, old in connection.execute(sa.select(table.c.id, table.c[column])).all():
            new = (
                json.dumps(_convert(json.loads(old), reverse), ensure_ascii=False)
                if column == "payload"
                else _convert(old, reverse)
            )
            if new != old:
                connection.execute(table.update().where(table.c.id == id).values({column: new}))


def upgrade():
    _rewrite()
    op.drop_column("sense_relations", "gloss")


def downgrade():
    # Removed glosses cannot be reconstructed; an empty value asserts no meaning.
    op.add_column("sense_relations", sa.Column("gloss", sa.Text, nullable=False, server_default=""))
    _rewrite(reverse=True)
