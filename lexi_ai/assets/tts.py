"""Text-to-speech seam: interface, stub, and OpenAI-compatible provider.

The ``TTSProvider`` protocol is the stable seam a real provider drops into.
``StubTTSProvider`` RAISES rather than returning empty bytes — a stub that
returned empty audio could be cached as if valid, poisoning the cache. Raising
keeps the cache clean (no row/file on a stubbed miss).

``OpenAICompatibleTTSProvider`` POSTs to an OpenAI-compatible ``/audio/speech``
endpoint via the ``openai`` SDK (no vendor TTS dep). Provider-specific config
(model, base_url, api_key) is passed in, not added to the protocol signature, so
the seam stays minimal.
"""

import ipaddress
from typing import Any, Protocol
from urllib.parse import urlparse


class TTSProvider(Protocol):
    async def synthesize(self, text: str, voice: str, fmt: str) -> bytes:
        """Synthesize ``text`` into audio bytes in ``fmt`` using ``voice``."""
        ...


class StubTTSProvider:
    """Not-yet-implemented provider. Raises so no bogus asset is ever cached."""

    async def synthesize(self, text: str, voice: str, fmt: str) -> bytes:
        raise NotImplementedError("TTS provider not configured — stub only")
        # real provider wires here


def _require_safe_base_url(base_url: str, api_key: str) -> None:
    """Reject a base_url that would ship ``api_key`` in cleartext.

    Only enforced when a key is set (no key → nothing to leak). Requires
    ``https://`` unless the host is loopback. An empty/malformed base_url with a
    key set is a config error, not a silent fallback — raise loudly.
    """
    if not api_key:
        return
    if not base_url:
        raise ValueError("a TTS base_url (LEXI_TTS_BASE_URL) is required when a TTS api_key is set")
    parsed = urlparse(base_url)
    host = parsed.hostname
    if not host:
        raise ValueError(f"malformed TTS base_url {base_url!r}: no host")
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and _is_loopback(host):
        return
    raise ValueError(
        f"non-https TTS base_url {base_url!r} would leak the api_key in cleartext; "
        "use https:// (or a loopback host for local testing)"
    )


def _is_loopback(host: str) -> bool:
    """Whether ``host`` is a loopback address — the literal ``localhost`` or any
    address in ``127.0.0.0/8`` / ``::1`` (so a plain-http URL never leaves the box)."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class OpenAICompatibleTTSProvider:
    """Synthesize speech via an OpenAI-compatible ``/audio/speech`` endpoint.

    A single POST (``audio.speech.create``) returns binary audio; we read it to
    bytes. The client is injectable for hermetic tests; the real one is built
    lazily from base_url/api_key so construction costs no network.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        client: Any | None = None,
    ):
        _require_safe_base_url(base_url, api_key)
        self._model = model
        if client is None:
            from openai import AsyncOpenAI

            # A keyless local compat server still needs a non-empty key to
            # construct the SDK client; it is never sent anywhere the guard did
            # not clear (loopback/https), and such servers ignore it.
            client = AsyncOpenAI(base_url=base_url, api_key=api_key or "not-needed")
        self._client: Any = client

    async def synthesize(self, text: str, voice: str, fmt: str) -> bytes:
        response = await self._client.audio.speech.create(
            model=self._model,
            input=text,
            voice=voice,
            response_format=fmt,
        )
        data = await response.aread()
        if not data:
            # A 200-with-empty-body (common on misconfigured compat servers) would
            # otherwise be cached as valid audio and served forever. Raise so the
            # miss is not persisted — same clean-cache posture as the stub.
            raise ValueError("TTS provider returned empty audio")
        return data
