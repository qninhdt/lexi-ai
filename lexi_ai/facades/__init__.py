"""The public API: two capability facades over the composition root.

Choose by what the caller is allowed to do, not by what it happens to need today.
:class:`LexiconReader` cannot mutate a row or reach a provider; :class:`LexiconEngine`
can do both. A reader deployment that never constructs the engine cannot spend a
model call by accident, which is the whole point of the split.
"""

from lexi_ai.facades.engine import LexiconEngine
from lexi_ai.facades.reader import LexiconReader

__all__ = ["LexiconEngine", "LexiconReader"]
