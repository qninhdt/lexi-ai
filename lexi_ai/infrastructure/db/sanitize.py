"""Text cleaning and column caps shared by every repository.

Every free-text value written here originates from an LLM or an external
reference, so it is untrusted. Two hazards drive this module:

* A NUL byte crashes a PostgreSQL INSERT, which would roll back a whole word.
* An embedded newline in a ``name``/``title`` breaks the topic vocab block that
  is re-injected into later prompts.

Caps annotated "must match" mirror a column width; the rest are generous sanity
bounds on unbounded ``Text`` columns.
"""

from lexi_ai.normalize import _CTRL_RE

MAX_TAG = 64
MAX_TITLE = 128
MAX_TAG_KEY = 255  # must match Tag.tag_key String(255) — NFKD can expand a key
MAX_GUIDEWORD = 64  # must match Sense.guideword String(64)
MAX_GLOSS = 255  # SenseRelation.gloss is Text (unbounded) — generous sanity cap only
MAX_IPA = 64  # must match Sense.ipa_uk/ipa_us String(64)
MAX_COLLOCATION = 512  # Collocation.text is Text (unbounded) — generous sanity cap only
MAX_SURFACE = 64  # SenseForm.surface is Text (unbounded) — generous sanity cap only
MAX_DOMAIN = 64  # must match Sense.domain String(64)
MAX_USAGE_NOTE = 255  # must match Sense.usage_note String(255)
MAX_THEME_NAME = 128  # Theme.name is Text (unbounded) — generous sanity cap only
MAX_THEME_KEY = 255  # must match Theme.theme_key String(255)
MAX_STYLE_PROMPT = 4000  # Theme.style_prompt is Text (unbounded) — generous sanity cap
MAX_THEME_DESCRIPTION = 1000
MAX_THEME_TONE = 255
MAX_THEMED_TEXT = 4000  # ThemedSense.definition / ThemedExample.text (Text) — generous cap
# Example.text is Text (unbounded). Appended examples MUST carry <t inf> tags, so
# the cap is generous (4000, matching themed) — a tight cap could sever a sentence
# mid-tag and hand parse_marked_example unbalanced markup.
MAX_EXAMPLE = 4000
MAX_DEFINITION = 4000  # Sense.definition is Text (unbounded) — generous sanity cap only
# norm / alias_norm are Text (unbounded) but feed match_key -> String(512); the
# schema already bounds the LLM inputs to 128 (GeneratedEntry/Alias.norm), so this
# generous cap only guards adversarial NFKD expansion, never a legit lemma.
MAX_NORM = 512
MAX_SOURCE_REF = 255  # must match SenseReference.source_ref String(255)


def clean(value: str, cap: int) -> str:
    """Single-line, control-free, trimmed, length-capped."""
    collapsed = _CTRL_RE.sub(" ", value)
    return " ".join(collapsed.split()).strip()[:cap]


def clean_opt(value: str | None, cap: int) -> str | None:
    """``clean`` for an OPTIONAL field: ``None``/empty in gives ``None`` out.

    A value that cleans away entirely (all control/whitespace) also collapses to
    ``None`` so the read side yields ``None``/``[]`` consistently.
    """
    if not value:
        return None
    return clean(value, cap) or None
