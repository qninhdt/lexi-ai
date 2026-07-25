"""The single home for ORM to domain to read-model translation.

Every conversion between a SQLAlchemy entity, a domain record, and a public read
model lives here. Keeping them together is what stops the copies from drifting:
the same view was previously assembled in several places and one copy silently
lost a field.
"""

from lexi_ai.domain.models import ThemeRecord, WordRecord
from lexi_ai.infrastructure.db.models import Theme, Word
from lexi_ai.read_models import Theme as ThemeView


def word_record(word: Word) -> WordRecord:
    """Detach a word row into a domain record."""
    return WordRecord(
        id=word.id,
        match_key=word.match_key,
        norm=word.norm,
        entry_type=word.entry_type,
        pos=word.pos,
        status=word.status,
        error_msg=word.error_msg,
    )


def theme_record(theme: Theme) -> ThemeRecord:
    """Detach a theme row into a domain record."""
    return ThemeRecord(
        id=theme.id,
        key=theme.theme_key,
        name=theme.name,
        style_prompt=theme.style_prompt,
        description=theme.description,
        tone=theme.tone,
    )


def theme_view(theme: ThemeRecord) -> ThemeView:
    """Project a theme record onto the public read model."""
    return ThemeView(
        key=theme.key,
        name=theme.name,
        style_prompt=theme.style_prompt,
        description=theme.description,
        tone=theme.tone,
    )
