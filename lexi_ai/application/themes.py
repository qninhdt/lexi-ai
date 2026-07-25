"""Style themes: managing voices and restyling an entry in one.

A theme is addressed by its normalized key, and that key is immutable once created,
so renaming a theme never re-keys it and callers keep the handle they already have.

Every provider call here stays outside the write transaction. Expanding a theme
profile or restyling a whole word is a model round trip; holding a write
transaction open across it would idle a connection for its duration.
"""

from collections.abc import Callable, Sequence

from lexi_ai.domain.ports import UnitOfWork
from lexi_ai.infrastructure.db.mappers import theme_view
from lexi_ai.normalize import theme_key as normalize_theme_key
from lexi_ai.read_models import SenseView, Theme


class ThemeService:
    """Theme use cases over the unit of work and the theming generators."""

    def __init__(
        self,
        uow_factory: Callable[[], UnitOfWork],
        themed_generator: Callable[[], object],
        metadata_generator: Callable[[], object],
        read_senses: Callable[[Sequence[int]], object],
        word_status: Callable[[int], object],
        max_examples_per_call: int,
    ) -> None:
        self._uow = uow_factory
        self._themed_generator = themed_generator
        self._metadata_generator = metadata_generator
        self._read_senses = read_senses
        self._word_status = word_status
        self._max_examples = max_examples_per_call

    async def create(
        self,
        name: str,
        style_prompt: str,
        description: str | None = None,
        tone: str | None = None,
    ) -> Theme:
        """Create or update a theme by its normalized key.

        With a description and tone supplied the theme is registered as given; with
        either missing, the model expands the name and prompt into a full profile
        first.
        """
        key = normalize_theme_key(name)
        if not key:
            raise ValueError(f"theme name yields no valid key: {name!r}")
        if description is None or tone is None:
            generated = await self._metadata_generator().generate(key, style_prompt)
            fields = {
                "name": generated.name,
                "style_prompt": generated.style_prompt,
                "description": generated.description,
                "tone": ",".join(generated.tone) if generated.tone else None,
            }
        else:
            fields = {
                "name": name,
                "style_prompt": style_prompt,
                "description": description,
                "tone": tone,
            }
        async with self._uow() as uow:
            theme = await uow.themes.create(key=key, overwrite=True, **fields)
            await uow.commit()
        return theme_view(theme)

    async def list_all(self) -> list[Theme]:
        """Every theme, name-sorted."""
        async with self._uow() as uow:
            themes = await uow.themes.list_all()
        return [theme_view(theme) for theme in themes]

    async def get(self, key: str) -> Theme | None:
        async with self._uow() as uow:
            theme = await uow.themes.get(normalize_theme_key(key))
        return theme_view(theme) if theme is not None else None

    async def update(
        self,
        key: str,
        *,
        name: str | None = None,
        style_prompt: str | None = None,
        description: str | None = None,
        tone: str | None = None,
    ) -> Theme:
        """Update an EXISTING theme; unset arguments are left alone.

        Unlike :meth:`create` this never creates, so an unknown key raises.
        """
        async with self._uow() as uow:
            theme = await uow.themes.update(
                normalize_theme_key(key),
                name=name,
                style_prompt=style_prompt,
                description=description,
                tone=tone,
            )
            await uow.commit()
        if theme is None:
            raise ValueError(f"unknown theme: {key!r}")
        return theme_view(theme)

    async def delete(self, key: str) -> bool:
        """Delete a theme. Its overlays cascade; neutral entries are untouched."""
        async with self._uow() as uow:
            deleted = await uow.themes.delete(normalize_theme_key(key))
            await uow.commit()
            return deleted

    async def resolve_or_raise(self, theme: str | int) -> tuple[int, str]:
        """Resolve a theme to ``(id, style_prompt)`` or raise."""
        resolved = await self.resolve(theme)
        if resolved is None:
            raise ValueError(f"unknown theme: {theme!r}")
        return resolved

    async def resolve(self, theme: str | int) -> tuple[int, str] | None:
        """Resolve a theme key or id, retrying once through the key normalizer.

        The repository is already key-first-then-id for a string; the retry only
        covers a caller passing an un-normalized display name.
        """
        async with self._uow() as uow:
            resolved = await uow.themes.resolve(theme)
            if resolved is None and isinstance(theme, str):
                resolved = await uow.themes.resolve(normalize_theme_key(theme))
            return resolved

    async def restyle_word(self, word_id: int, theme_id: int, style_prompt: str) -> None:
        """Restyle every sense of a done word in one voice.

        Only a done word can be themed: restyling depends on the neutral senses
        existing, so a pending or errored word is a caller mistake rather than
        something to wait for.
        """
        status = await self._word_status(word_id)
        if status != "done":
            raise ValueError(f"word {word_id} is not done (status={status!r})")
        async with self._uow() as uow:
            neutral = await uow.senses.for_theming(word_id)
        if not neutral:
            raise ValueError(f"word {word_id} has no senses to theme")
        sense_ids = [row.sense_id for row in neutral]
        facts = [(row.definition, row.pos, row.guideword, row.tier) for row in neutral]
        result = await self._themed_generator().generate(style_prompt, facts)
        async with self._uow() as uow:
            # The ordering is the contract: themed result i maps to sense_ids[i].
            await uow.themes.persist_themed(theme_id, result, sense_ids)
            await uow.commit()

    async def sense_view(self, sense_id: int, theme_id: int) -> SenseView:
        """One sense with its themed definition and examples overlaid."""
        base = (await self._read_senses([sense_id]))[0]
        async with self._uow() as uow:
            word_id = await uow.senses.word_id_for(sense_id)
            overlay = await uow.themes.overlay_for_word(word_id, theme_id)
        themed = overlay.get(sense_id)
        if themed is not None:
            base.definition, base.examples = themed
        return base

    async def append_examples(self, sense_id: int, n: int, theme: str | int) -> SenseView:
        """Append up to ``n`` in-voice examples to a sense's themed overlay.

        The overlay must already exist (the word themed via generate): a missing
        theme, or a sense with no themed row for that theme, raises rather than
        silently theming the whole word as a side effect of asking for examples.
        """
        theme_id, style_prompt = await self.resolve_or_raise(theme)
        async with self._uow() as uow:
            context = await uow.senses.example_context(sense_id)
            if context is None:
                raise ValueError(f"unknown sense_id: {sense_id}")
            overlay = await uow.themes.overlay_for_sense(sense_id, theme_id)
        if overlay is None:
            raise ValueError(
                f"sense {sense_id} has no themed overlay for theme {theme!r}; "
                "theme the word first via generate(theme=)"
            )
        themed_sense_id, existing_themed = overlay
        facts, _neutral_examples = context
        n = min(n, self._max_examples)
        if n > 0:
            batch = await self._themed_generator().generate_examples(
                style_prompt, facts, existing_themed, n
            )
            async with self._uow() as uow:
                await uow.themes.append_themed_examples(themed_sense_id, batch.examples)
                await uow.commit()
        return await self.sense_view(sense_id, theme_id)
