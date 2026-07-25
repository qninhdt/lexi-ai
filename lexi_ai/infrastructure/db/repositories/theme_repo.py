"""The ``themes`` aggregate: style voices and the themed overlay of a sense."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from sqlalchemy import CursorResult, delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from lexi_ai.domain.models import ThemeRecord
from lexi_ai.infrastructure.db.mappers import theme_record
from lexi_ai.infrastructure.db.models import Sense, Theme, ThemedExample, ThemedSense
from lexi_ai.infrastructure.db.sanitize import (
    MAX_STYLE_PROMPT,
    MAX_THEME_DESCRIPTION,
    MAX_THEME_KEY,
    MAX_THEME_NAME,
    MAX_THEME_TONE,
    MAX_THEMED_TEXT,
    clean,
)
from lexi_ai.normalize import theme_key

if TYPE_CHECKING:
    from lexi_ai.theming.schemas import ThemedResult


class SqlThemeRepo:
    """Session-bound implementation of :class:`lexi_ai.domain.ports.ThemeRepo`.

    Reads return :class:`ThemeRecord`, not the mapped row, so a caller can hold a
    theme after its transaction closed without depending on session state.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        name: str,
        style_prompt: str,
        description: str | None = None,
        tone: str | None = None,
        key: str | None = None,
        overwrite: bool = False,
    ) -> ThemeRecord:
        """Resolve-or-create a theme by key, optionally overwriting its fields.

        A concurrent create of the same key is recovered by re-fetching inside a
        SAVEPOINT so the outer transaction survives.
        """
        final_key = theme_key(key) if key is not None else theme_key(name)
        if not final_key or len(final_key) > MAX_THEME_KEY:
            raise ValueError(f"theme key/name yields no valid key: {key or name!r}")
        fields = self._clean_fields(name, style_prompt, description, tone)
        existing = await self._get(final_key)
        if existing is not None:
            if overwrite:
                self._overwrite(existing, fields)
            return theme_record(existing)
        theme = Theme(theme_key=final_key, **fields)
        try:
            async with self._session.begin_nested():
                self._session.add(theme)
                await self._session.flush()
        except IntegrityError:
            existing = await self._get(final_key)
            if existing is None:
                raise
            if overwrite:
                self._overwrite(existing, fields)
            return theme_record(existing)
        return theme_record(theme)

    async def list_all(self) -> list[ThemeRecord]:
        """Every theme, name-sorted."""
        rows = await self._session.execute(select(Theme).order_by(Theme.name))
        return [theme_record(theme) for theme in rows.scalars()]

    async def get(self, key: str) -> ThemeRecord | None:
        theme = await self._get(key)
        return theme_record(theme) if theme is not None else None

    async def update(
        self,
        key: str,
        name: str | None = None,
        style_prompt: str | None = None,
        description: str | None = None,
        tone: str | None = None,
    ) -> ThemeRecord | None:
        """Partially update an EXISTING theme; never creates.

        ``theme_key`` is immutable, so renaming never re-keys the theme and callers
        keep addressing it by the same key. Returns ``None`` when the key is
        unknown.
        """
        existing = await self._get(key)
        if existing is None:
            return None
        if name is not None:
            existing.name = clean(name, MAX_THEME_NAME)
        if style_prompt is not None:
            existing.style_prompt = clean(style_prompt, MAX_STYLE_PROMPT)
        if description is not None:
            existing.description = clean(description, MAX_THEME_DESCRIPTION)
        if tone is not None:
            existing.tone = clean(tone, MAX_THEME_TONE)
        await self._session.flush()
        return theme_record(existing)

    async def delete(self, key: str) -> bool:
        """Delete a theme by key. Cascades its overlays; neutral entries are untouched."""
        result = await self._session.execute(delete(Theme).where(Theme.theme_key == key))
        return (cast("CursorResult", result).rowcount or 0) > 0

    async def resolve(self, key_or_id: str | int) -> tuple[int, str] | None:
        """``(theme_id, style_prompt)`` for a key or id, else ``None``.

        An ``int`` is always an id. A ``str`` tries the key FIRST and falls back to
        an id lookup only on a key miss, so a theme literally named "1984" stays
        addressable by name while ``?theme=42`` still resolves by id.
        """
        if isinstance(key_or_id, int):
            return await self._resolve_by_id(key_or_id)
        found = await self._select_theme(Theme.theme_key == theme_key(key_or_id))
        if found is not None:
            return found
        try:
            theme_id = int(key_or_id)
        except ValueError:
            return None
        return await self._resolve_by_id(theme_id)

    async def persist_themed(
        self, theme_id: int, result: "ThemedResult", sense_ids: Sequence[int]
    ) -> None:
        """Overwrite the themed rows for ``(sense_ids, theme_id)`` in place.

        ``sense_ids`` arrives in the same order the prompt numbered the senses, so
        ``result.senses[i]`` maps to ``sense_ids[i]``. A count mismatch is a hard
        error rather than a silent zip: the model returned the wrong number of
        senses. Core delete plus explicit-FK insert only, never a relationship
        collection on a persistent object.
        """
        if len(result.senses) != len(sense_ids):
            raise ValueError(
                f"themed sense count {len(result.senses)} != neutral sense count {len(sense_ids)}"
            )
        for sense_id, themed in zip(sense_ids, result.senses, strict=True):
            await self._session.execute(
                delete(ThemedSense).where(
                    ThemedSense.sense_id == sense_id,
                    ThemedSense.theme_id == theme_id,
                )
            )
            row = ThemedSense(
                sense_id=sense_id,
                theme_id=theme_id,
                definition=clean(themed.definition, MAX_THEMED_TEXT),
            )
            self._session.add(row)
            await self._session.flush()
            order = 0
            for example in themed.examples:
                text = clean(example, MAX_THEMED_TEXT)
                if not text:
                    continue
                self._session.add(
                    ThemedExample(themed_sense_id=row.id, text=text, example_order=order)
                )
                order += 1
        await self._session.flush()

    async def overlay_for_word(
        self, word_id: int, theme_id: int
    ) -> dict[int, tuple[str, list[str]]]:
        """``{sense_id: (themed_definition, [themed_examples])}`` for one word.

        One companion query rather than one per sense. Senses without a themed row
        are simply absent and the read layer falls back to neutral.
        """
        rows = await self._session.execute(
            select(ThemedSense.id, ThemedSense.sense_id, ThemedSense.definition)
            .join(Sense, Sense.id == ThemedSense.sense_id)
            .where(Sense.word_id == word_id, ThemedSense.theme_id == theme_id)
        )
        themed = list(rows.all())
        if not themed:
            return {}
        example_rows = await self._session.execute(
            select(ThemedExample.themed_sense_id, ThemedExample.text)
            .where(ThemedExample.themed_sense_id.in_([row[0] for row in themed]))
            .order_by(ThemedExample.example_order)
        )
        examples: dict[int, list[str]] = {}
        for themed_sense_id, text in example_rows:
            examples.setdefault(themed_sense_id, []).append(text)
        return {
            sense_id: (definition, examples.get(themed_sense_id, []))
            for themed_sense_id, sense_id, definition in themed
        }

    async def overlay_for_sense(self, sense_id: int, theme_id: int) -> tuple[int, list[str]] | None:
        """``(themed_sense_id, ordered_example_texts)`` for one overlay, else ``None``.

        ``None`` is the "theme the word first" signal: the caller raises rather
        than silently theming the whole word.
        """
        row = (
            await self._session.execute(
                select(ThemedSense.id).where(
                    ThemedSense.sense_id == sense_id,
                    ThemedSense.theme_id == theme_id,
                )
            )
        ).first()
        if row is None:
            return None
        themed_sense_id = row[0]
        texts = (
            (
                await self._session.execute(
                    select(ThemedExample.text)
                    .where(ThemedExample.themed_sense_id == themed_sense_id)
                    .order_by(ThemedExample.example_order)
                )
            )
            .scalars()
            .all()
        )
        return themed_sense_id, list(texts)

    async def append_themed_examples(self, themed_sense_id: int, texts: Sequence[str]) -> int:
        """Append cleaned, non-empty texts after the current highest order.

        Never overwrites existing themed examples, unlike the whole-word path.
        """
        current_max = (
            await self._session.execute(
                select(func.max(ThemedExample.example_order)).where(
                    ThemedExample.themed_sense_id == themed_sense_id
                )
            )
        ).scalar_one_or_none()
        order = (current_max + 1) if current_max is not None else 0
        inserted = 0
        for text in texts:
            cleaned = clean(text, MAX_THEMED_TEXT)
            if not cleaned:
                continue
            self._session.add(
                ThemedExample(themed_sense_id=themed_sense_id, text=cleaned, example_order=order)
            )
            order += 1
            inserted += 1
        await self._session.flush()
        return inserted

    @staticmethod
    def _clean_fields(
        name: str, style_prompt: str, description: str | None, tone: str | None
    ) -> dict[str, str | None]:
        return {
            "name": clean(name, MAX_THEME_NAME),
            "style_prompt": clean(style_prompt, MAX_STYLE_PROMPT),
            "description": clean(description, MAX_THEME_DESCRIPTION) if description else None,
            "tone": clean(tone, MAX_THEME_TONE) if tone else None,
        }

    @staticmethod
    def _overwrite(theme: Theme, fields: dict[str, str | None]) -> None:
        theme.name = cast(str, fields["name"])
        theme.style_prompt = cast(str, fields["style_prompt"])
        theme.description = fields["description"]
        theme.tone = fields["tone"]

    async def _resolve_by_id(self, theme_id: int) -> tuple[int, str] | None:
        return await self._select_theme(Theme.id == theme_id)

    async def _select_theme(self, condition) -> tuple[int, str] | None:  # noqa: ANN001
        row = (
            await self._session.execute(select(Theme.id, Theme.style_prompt).where(condition))
        ).first()
        return (row[0], row[1]) if row is not None else None

    async def _get(self, key: str) -> Theme | None:
        result = await self._session.execute(select(Theme).where(Theme.theme_key == key))
        return result.scalar_one_or_none()
