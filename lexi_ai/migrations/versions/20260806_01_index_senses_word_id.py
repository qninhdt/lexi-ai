"""index senses.word_id

``senses.word_id`` is the most-filtered column in the repository: every read of a
word's senses and the join behind entry assembly select on it, across eight call
sites. It had no index. Unlike ``word_aliases`` and ``word_tags`` — whose UNIQUE
constraints lead with ``word_id`` and are therefore already backed by a usable
btree — ``senses`` declares no such constraint, so Postgres had no access path
but a sequential scan of the whole table, paid once per entry read.

Built with a plain ``CREATE INDEX``, not ``CONCURRENTLY``. The concurrent form
cannot run here: ``env.py`` opens one transaction around the whole upgrade
(``connectable.begin()`` + ``run_sync``), and CONCURRENTLY is rejected inside a
transaction block — Alembic's ``autocommit_block`` asserts rather than escaping an
async connection's transaction. A plain build takes an exclusive lock on
``senses`` for its duration, which is acceptable at this table's size and matches
how every other index in this schema was created.

Revision ID: 20260806_01
Revises: 20260724_02
Create Date: 2026-08-06
"""

from alembic import op

revision = "20260806_01"
down_revision = "20260724_02"
branch_labels = None
depends_on = None

_INDEX = "ix_senses_word_id"
_TABLE = "senses"
_COLUMN = "word_id"


def _schema() -> str:
    """The configured domain schema, resolved by the alembic environment."""
    from alembic import context

    return context.config.attributes["lexi_schema"]


def upgrade() -> None:
    op.create_index(_INDEX, _TABLE, [_COLUMN], schema=_schema())


def downgrade() -> None:
    op.drop_index(_INDEX, table_name=_TABLE, schema=_schema())
