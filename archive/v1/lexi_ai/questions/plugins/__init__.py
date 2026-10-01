"""The five built-in MVP question types.

Importing this package runs each module's ``register()`` call.
"""

from lexi_ai.questions.plugins.cloze import Cloze
from lexi_ai.questions.plugins.contextual_mcq import ContextualMCQ
from lexi_ai.questions.plugins.definition_mcq import DefinitionMCQ
from lexi_ai.questions.plugins.flashcard import Flashcard
from lexi_ai.questions.plugins.use_in_sentence import UseInSentence

__all__ = [
    "Cloze",
    "ContextualMCQ",
    "DefinitionMCQ",
    "Flashcard",
    "UseInSentence",
]
