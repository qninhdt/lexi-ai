"""Format plugins — one module per format, each self-registering on import.

Importing this package imports every format module below, and each module runs
its own ``register(FormatSpec(...))`` at module scope — so the registry is
populated as a side effect of importing the package (the same contract the old
single ``formats.py`` had). Cross-format helpers live in ``_shared``; no format
module imports another.

The plugin classes are re-exported here so existing call sites
(``from lexi_ai.questions.formats import Cloze``) keep working after the split.
"""

from lexi_ai.questions.formats.cloze import Cloze
from lexi_ai.questions.formats.collocation_fill import CollocationFill
from lexi_ai.questions.formats.contextual_mcq import ContextualMCQ
from lexi_ai.questions.formats.definition_mcq import DefinitionMCQ
from lexi_ai.questions.formats.listening import Listening
from lexi_ai.questions.formats.matching import Matching
from lexi_ai.questions.formats.pronunciation_mcq import PronunciationMCQ
from lexi_ai.questions.formats.spelling import Spelling
from lexi_ai.questions.formats.use_in_sentence import UseInSentence

__all__ = [
    "Cloze",
    "CollocationFill",
    "ContextualMCQ",
    "DefinitionMCQ",
    "Listening",
    "Matching",
    "PronunciationMCQ",
    "Spelling",
    "UseInSentence",
]
