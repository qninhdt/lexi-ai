"""The five built-in MVP question types.

Importing this package runs each module's ``register()`` call, populating the
registry by direct import.
"""

from lexi_ai.questions.types.cloze import Cloze
from lexi_ai.questions.types.contextual_mcq import ContextualMCQ
from lexi_ai.questions.types.definition_mcq import DefinitionMCQ
from lexi_ai.questions.types.flashcard import Flashcard
from lexi_ai.questions.types.use_in_sentence import UseInSentence

__all__ = [
    "Cloze",
    "ContextualMCQ",
    "DefinitionMCQ",
    "Flashcard",
    "UseInSentence",
]
