"""Private Cambridge storage and reference artifact import markers."""

from sqlalchemy import Column, Integer, MetaData, Table, Text

SCHEMA = "lexi_reference"
metadata = MetaData(schema=SCHEMA)

words = Table(
    "words",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=False),
    Column("word", Text),
    Column("display_form", Text),
    Column("entry_type", Text),
    Column("status", Text),
)
entries = Table(
    "entries",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=False),
    Column("word_id", Integer, nullable=False, index=True),
    Column("pos", Text),
    Column("headword", Text),
    Column("entry_order", Integer),
    Column("pronunciation_uk", Text),
    Column("pronunciation_us", Text),
)
senses = Table(
    "senses",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=False),
    Column("entry_id", Integer, nullable=False, index=True),
    Column("definition", Text),
    Column("cefr_level", Text),
    Column("phrase_title", Text),
    Column("sense_order", Integer),
)
entry_inflections = Table(
    "entry_inflections",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=False),
    Column("entry_id", Integer, nullable=False, index=True),
    Column("form_type", Text),
    Column("inflected_form", Text, nullable=False),
)
examples = Table(
    "examples",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=False),
    Column("sense_id", Integer, nullable=False, index=True),
    Column("example", Text),
    Column("example_order", Integer),
)
word_alternatives = Table(
    "word_alternatives",
    metadata,
    Column("word_id", Integer, primary_key=True),
    Column("alternative_word", Text, primary_key=True),
    Column("alternative_type", Text),
)
datasets = Table(
    "datasets",
    metadata,
    Column("name", Text, primary_key=True),
    Column("checksum", Text, nullable=False),
)

CAMBRIDGE_TABLES = (words, entries, entry_inflections, senses, examples, word_alternatives)
