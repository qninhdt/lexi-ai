"""Column types that enforce the controlled vocabularies at the storage layer.

Most of these columns used to be plain strings whose only guard was the LLM output
schema. That guard covers the generation path and nothing else, so any other
writer — a reference importer, a stub row, a hand-written fix — could store a
value outside the vocabulary. A drifted value is quiet damage: the sense-relation
part-of-speech filter compares labels, so a stored ``adj`` where ``adjective`` is
expected silently mis-filters candidates instead of failing.

Validation therefore belongs to the column, which every writer goes through.

The implementation stays ``String`` rather than a native enum: the schema has to
run identically on SQLite and PostgreSQL, and a native enum would need its own
migration to add a member.
"""

from collections.abc import Collection

from sqlalchemy import String
from sqlalchemy.types import TypeDecorator


class Vocabulary(TypeDecorator):
    """A string column restricted to a controlled vocabulary.

    ``None`` passes through for nullable columns; the vocabulary describes which
    values are meaningful, not whether one is required.
    """

    impl = String
    cache_ok = True

    def __init__(self, length: int, vocabulary: Collection[str], label: str) -> None:
        super().__init__(length=length)
        self._vocabulary = frozenset(vocabulary)
        self._label = label

    def process_bind_param(self, value: str | None, dialect) -> str | None:  # noqa: ANN001
        if value is None:
            return None
        if value not in self._vocabulary:
            raise ValueError(
                f"{self._label}: {value!r} is not in the controlled vocabulary "
                f"({', '.join(sorted(self._vocabulary))})"
            )
        return value

    def copy(self, **kwargs) -> "Vocabulary":  # noqa: ANN003
        return Vocabulary(self.impl.length, self._vocabulary, self._label)


class VocabularyList(TypeDecorator):
    """A comma-joined set of vocabulary tokens, exposed as a list.

    The column stores several labels at once and is never queried on its own, so a
    joined string is enough and keeps the schema portable. Doing the join and split
    here means callers hold a list on both sides and no read site has to remember
    the encoding — that split logic was previously duplicated at every reader.

    A label may not contain the separator; a test asserts that for the vocabulary,
    and this rejects it for anything else.
    """

    impl = String
    cache_ok = True
    _SEPARATOR = ","

    def __init__(self, length: int, vocabulary: Collection[str], label: str) -> None:
        super().__init__(length=length)
        self._vocabulary = frozenset(vocabulary)
        self._label = label

    def process_bind_param(self, value: list[str] | None, dialect) -> str | None:  # noqa: ANN001
        if not value:
            # Empty and unset collapse to NULL so the read side always yields [].
            return None
        for token in value:
            if token not in self._vocabulary:
                raise ValueError(
                    f"{self._label}: {token!r} is not in the controlled vocabulary "
                    f"({', '.join(sorted(self._vocabulary))})"
                )
            if self._SEPARATOR in token:
                raise ValueError(f"{self._label}: {token!r} contains the list separator")
        return self._SEPARATOR.join(value)

    def process_result_value(self, value: str | None, dialect) -> list[str]:  # noqa: ANN001
        if not value:
            return []
        return value.split(self._SEPARATOR)

    def copy(self, **kwargs) -> "VocabularyList":  # noqa: ANN003
        return VocabularyList(self.impl.length, self._vocabulary, self._label)
