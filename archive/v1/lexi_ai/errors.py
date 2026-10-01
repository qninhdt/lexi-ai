

class StaleGenerationError(RuntimeError):
    """A newer generation claim superseded this worker before it could publish."""


class ThemeError(ValueError):
    """A theme operation cannot proceed. Base for the specific causes below.

    Subclasses :class:`ValueError` so a caller that wants to distinguish causes
    branches on the subclasses instead of parsing messages.
    """


class UnknownTheme(ThemeError):
    """No theme matches the requested key or id."""


class InvalidThemeName(ThemeError):
    """A proposed theme name normalizes to no usable key, so none can be created."""


class ThemeTargetInvalid(ThemeError):
    """The word or sense a theme operation targets is missing or not themeable
    (only a done word with neutral senses can be restyled).
    """


class ThemedOverlayMissing(ThemeError):
    """The sense has no themed overlay for the requested theme; the word must be
    themed first via ``generate(theme=...)``.
    """
