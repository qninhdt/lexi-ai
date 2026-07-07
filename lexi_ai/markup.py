"""Example target markup — the single reader for the ``<t inf="...">`` tags.

The generation prompt REQUIRES the model to wrap the target word/phrase in every
example sentence with ``<t inf="label">surface</t>`` (see
``prompts/senses_generation_system.jinja``). The tag is a deliberate design
choice, NOT noise: it marks the target for display highlighting and tells the
cloze plugin exactly which span to blank. Examples are stored WITH the tags
intact; this module is the one place that reads them, so no consumer reinvents
the regex (mirrors ``normalize.match_key`` being the one key function).

Robust to un-tagged input: a plain string (e.g. a fake-LLM test fixture) parses
to itself with an empty span list, so nothing downstream breaks when tags are
absent.
"""

import re
from dataclasses import dataclass

__all__ = ["Span", "parse_marked_example", "strip_markup"]

# One target tag: <t inf="past">glistened</t>. ``inf`` is bounded to word chars
# (the closed INFLECTION_LABELS vocab); the inner surface is captured lazily so a
# sentence with several tags does not span across them.
_TAG_RE = re.compile(r'<t\s+inf="([^"]*)"\s*>(.*?)</t>', re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class Span:
    """A marked target occurrence, positioned in the RENDERED (tag-free) text.

    ``surface`` is the inflected form as it appears in the sentence; ``inf`` is
    its inflection label (∈ INFLECTION_LABELS, though this module does not
    validate the vocab — it only reports what the tag carried). ``start``/``end``
    index into the clean text returned alongside, so a caller can blank exactly
    ``clean[start:end]``.
    """

    surface: str
    inf: str
    start: int
    end: int


def parse_marked_example(text: str) -> tuple[str, list[Span]]:
    """Split a marked example into (clean_text, spans).

    ``clean_text`` is the sentence with every ``<t>`` tag unwrapped to its inner
    surface (display form). ``spans`` locate each target within ``clean_text``.
    Un-tagged input returns unchanged with no spans.
    """
    spans: list[Span] = []
    out: list[str] = []
    pos = 0  # cursor into the ORIGINAL text
    clean_len = 0  # running length of the clean text built so far
    for m in _TAG_RE.finditer(text):
        out.append(text[pos : m.start()])
        clean_len += m.start() - pos
        inf = m.group(1)
        surface = m.group(2)
        out.append(surface)
        spans.append(Span(surface=surface, inf=inf, start=clean_len, end=clean_len + len(surface)))
        clean_len += len(surface)
        pos = m.end()
    out.append(text[pos:])
    return "".join(out), spans


def strip_markup(text: str) -> str:
    """The example's display form: tags unwrapped to their inner surface."""
    return parse_marked_example(text)[0]
