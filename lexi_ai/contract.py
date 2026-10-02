"""Public database integration contract for embedded/shared-database consumers."""

from sqlalchemy import Integer, MetaData, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

LEXI_SCHEMA: str = "lexi"
SENSE_PRIMARY_KEY: str = "id"


class RelationalContractBase(DeclarativeBase):
    metadata = MetaData(schema=LEXI_SCHEMA)


class Sense(RelationalContractBase):
    """Public Sense model representing the lexi.senses relational contract for consumers.

    Consumers may use this model for direct relational read composition (JOIN,
    ORDER BY, CEFR filter, pagination) in PostgreSQL when sharing a database with Lexi.
    """

    __tablename__ = "senses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    word_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    pos: Mapped[str] = mapped_column(String(32), nullable=False)
    tier: Mapped[str] = mapped_column(String(16), nullable=False)
    cefr_level: Mapped[str | None] = mapped_column(String(8))
    register: Mapped[str | None] = mapped_column(String(32))
    usage_note: Mapped[str | None] = mapped_column(Text)
    ipa_uk: Mapped[str | None] = mapped_column(String(80))
    ipa_us: Mapped[str | None] = mapped_column(String(80))


metadata = RelationalContractBase.metadata

__all__ = [
    "LEXI_SCHEMA",
    "SENSE_PRIMARY_KEY",
    "Sense",
    "metadata",
]
