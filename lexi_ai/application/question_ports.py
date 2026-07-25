"""Adapters that hand the question engine the two capabilities it cannot own.

The engine must not know about the dictionary or the asset cache — it only needs
"resolve this sense to its entry" and "make me a clip for this text". Both arrive
as narrow ports so a question plugin stays decoupled from persistence and config.
"""

from collections.abc import Awaitable, Callable

from sqlalchemy.exc import NoResultFound

from lexi_ai.domain.ports import UnitOfWork
from lexi_ai.read_models import Entry


class SenseEntryLoader:
    """Resolves a sense to its owning entry, for provider-free exposure cards."""

    def __init__(
        self,
        uow_factory: Callable[[], UnitOfWork],
        read_entry: Callable[[int], Awaitable[Entry]],
    ) -> None:
        self._uow = uow_factory
        self._read_entry = read_entry

    async def load_entry(self, sense_id: int) -> Entry | None:
        """The sense's owning entry, or ``None`` when the sense is gone."""
        async with self._uow() as uow:
            try:
                word_id = await uow.senses.word_id_for(sense_id)
            except NoResultFound:
                return None
        return await self._read_entry(word_id)


class AssetTtsPort:
    """Adapts the cached-asset TTS surface to the questions ``TtsPort`` seam.

    ``ensure_clip`` synthesizes cache-first and returns the clip's
    ``(source_kind, source_id, voice, fmt)`` reference tuple — never a row id, so a
    frozen question payload survives a purge/regenerate. Voice and format are
    resolved per call by the supplied callable, keeping plugins out of config and
    honouring a settings change made after the engine was built.

    Returns ``None`` when no real clip can be made (the source text is gone, or an
    empty-text short circuit), so the audio formats degrade instead of fabricating
    an asset.
    """

    def __init__(
        self,
        speak: Callable[[str, int, str, str], Awaitable[object]],
        voice_and_format: Callable[[], tuple[str, str]],
    ) -> None:
        self._speak = speak
        self._voice_and_format = voice_and_format

    async def ensure_clip(
        self, source_kind: str, source_id: int
    ) -> tuple[str, int, str, str] | None:
        voice, fmt = self._voice_and_format()
        try:
            asset = await self._speak(source_kind, source_id, voice, fmt)
        except ValueError:
            return None  # no source text for the ref — nothing to synthesize
        if not asset.ready:
            return None  # empty/whitespace source short-circuited — no real clip
        return (source_kind, source_id, voice, fmt)
