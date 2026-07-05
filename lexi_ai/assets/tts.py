"""Text-to-speech seam (Phase 6): interface + stub only.

The ``TTSProvider`` protocol is the stable seam a real provider drops into later.
This round ships ``StubTTSProvider``, which RAISES rather than returning empty
bytes — a stub that returned empty audio could be cached as if valid, poisoning
the cache. Raising keeps the cache clean (no row/file on a stubbed miss). Anything
provider-specific (model, base_url) is read from settings inside the provider, not
added to the protocol signature, so the seam stays minimal.
"""

from typing import Protocol


class TTSProvider(Protocol):
    async def synthesize(self, text: str, voice: str, fmt: str) -> bytes:
        """Synthesize ``text`` into audio bytes in ``fmt`` using ``voice``."""
        ...


class StubTTSProvider:
    """Not-yet-implemented provider. Raises so no bogus asset is ever cached."""

    async def synthesize(self, text: str, voice: str, fmt: str) -> bytes:
        raise NotImplementedError("TTS provider not configured — stub only")
        # real provider wires here
