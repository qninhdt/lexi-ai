"""How an asset is identified, and how its source text is verified.

Pure functions over strings and the closed vocabularies in `constants.py`: no
session, no table, no filesystem. They lived in the asset repository because that
is where they were first needed, which put the identity rules — the thing every
caller must agree on — inside a module the application layer is forbidden to
import.

Both run on read AND write, and that symmetry is the contract. `content_hash`
normalizes before hashing so whitespace variants of one text collapse to one hash;
`normalize_asset_params` validates every free parameter against a closed vocabulary
at a single choke point, which is what closed the collision where `en-US` and
`en_US` squashed to the same on-disk path and served each other's bytes.
"""

import hashlib

from lexi_ai.config import get_settings
from lexi_ai.constants import (
    ASSET_KINDS,
    TRANSLATION_LANGUAGES,
    TTS_FORMATS,
    TTS_VOICES,
)
from lexi_ai.normalize import _CTRL_RE


def content_hash(text: str) -> str:
    """sha256 hex of the NORMALIZED source text (VERIFY function, not identity).

    Normalization (strip control chars, collapse whitespace, strip) runs before
    hashing so trailing/interior-whitespace variants of the same text collapse to
    one hash. The SAME normalization on every call — this is the verify contract.
    """
    s = _CTRL_RE.sub(" ", text)
    s = " ".join(s.split()).strip()
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def normalize_asset_params(kind: str, **kw: str | None) -> str:
    """Stable param token for an asset identity, normalized on read AND write.

    ``translate`` → a normalized lang code (``lang``); ``tts`` → ``voice|fmt``.
    Unknown kind → ``ValueError``.

    Every free param is validated against a closed vocab at this one choke point
    (like ``lang`` against ``TRANSLATION_LANGUAGES``): ``voice``/``fmt`` against
    ``TTS_VOICES``/``TTS_FORMATS``. This closes the filename-collision bug where
    two distinct DB rows (``en-US`` vs ``en_US``) squashed to the SAME on-disk
    path and served each other's bytes. A ``None`` voice/fmt resolves to the
    configured default (``alloy``/``mp3``) BEFORE validation, so a default TTS
    call never hard-rejects on the happy path.
    """
    if kind not in ASSET_KINDS:
        raise ValueError(f"unknown asset kind: {kind!r}")
    if kind == "translate":
        lang = _norm_token(kw.get("lang"))
        if lang not in TRANSLATION_LANGUAGES:
            raise ValueError(f"invalid/unsupported language code: {lang!r}")
        return lang
    # tts — resolve None to the configured default, then validate both params.
    settings = get_settings()
    voice = _norm_token(kw.get("voice") if kw.get("voice") is not None else settings.tts_voice)
    fmt = _norm_token(kw.get("fmt") if kw.get("fmt") is not None else settings.tts_format)
    if voice not in TTS_VOICES:
        raise ValueError(f"invalid/unsupported TTS voice: {voice!r}")
    if fmt not in TTS_FORMATS:
        raise ValueError(f"invalid/unsupported TTS format: {fmt!r}")
    return f"{voice}|{fmt}"


def _norm_token(value: str | None) -> str:
    return (value or "").strip().lower()
