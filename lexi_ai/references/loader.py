"""Reference loader (Phase 3): merge Cambridge + WordNet into a bundle.

The :class:`ReferenceBundle` is the hallucination anchor handed to the LLM
prompt in Phase 4. WordNet content may be empty (idioms/expressions) — that is
acceptable (decision #12), not an error.
"""

from dataclasses import dataclass, field

from lexi_ai.references.cambridge import CambridgeEntry, CambridgeSource, CamSense
from lexi_ai.references.wordnet import WnSense, WordNetSource


@dataclass
class ReferenceBundle:
    word_raw: str
    entry_type: str | None
    cambridge_word_id: int | None
    cambridge_senses: list[CamSense] = field(default_factory=list)
    cambridge_alternatives: list[tuple[str, str]] = field(default_factory=list)
    wordnet_synsets: list[WnSense] = field(default_factory=list)

    @property
    def has_wordnet(self) -> bool:
        return bool(self.wordnet_synsets)

    @property
    def has_cambridge(self) -> bool:
        return self.cambridge_word_id is not None


class ReferenceLoader:
    """Combine the Cambridge and WordNet sources for one lookup word."""

    def __init__(self, cambridge: CambridgeSource, wordnet: WordNetSource):
        self._cambridge = cambridge
        self._wordnet = wordnet

    @property
    def cambridge(self) -> CambridgeSource:
        """The Cambridge source, for surface resolution (Lexicon.resolve)."""
        return self._cambridge

    async def bundle(self, word: str) -> ReferenceBundle:
        cam = await self._cambridge.fetch(word)

        if cam is None:
            # No Cambridge anchor — still try WordNet on the raw input.
            synsets = await self._wordnet.lookup(word)
            return ReferenceBundle(
                word_raw=word,
                entry_type=None,
                cambridge_word_id=None,
                wordnet_synsets=synsets,
            )
        return await self._bundle_from_cambridge(cam)

    async def bundle_by_id(self, cambridge_word_id: int) -> ReferenceBundle | None:
        """Build a bundle for a resolved Cambridge word_id (skips re-matching).

        Returns ``None`` only if the id vanished from the source (it should not for
        an id that came from :meth:`CambridgeSource.resolve_exact`).
        """
        cam = await self._cambridge.fetch_by_id(cambridge_word_id)
        if cam is None:
            return None
        return await self._bundle_from_cambridge(cam)

    async def bundle_custom(self, word: str) -> ReferenceBundle:
        """Build a WordNet-only bundle for a custom (non-Cambridge) word.

        Forces no Cambridge anchor even if the surface happens to exist in
        Cambridge — ``add`` is an explicit "not from Cambridge" request.
        """
        synsets = await self._wordnet.lookup(word)
        return ReferenceBundle(
            word_raw=word,
            entry_type=None,
            cambridge_word_id=None,
            wordnet_synsets=synsets,
        )

    async def _bundle_from_cambridge(self, cam: CambridgeEntry) -> ReferenceBundle:
        # Distinct POS present in Cambridge senses, to steer WordNet lookup.
        cam_pos = {s.pos for s in cam.senses if s.pos}
        synsets: list[WnSense] = []
        if cam_pos:
            for pos in cam_pos:
                synsets.extend(await self._wordnet.lookup(cam.display_form, pos))
        else:
            synsets = await self._wordnet.lookup(cam.display_form)

        return ReferenceBundle(
            word_raw=cam.display_form,
            entry_type=cam.entry_type,
            cambridge_word_id=cam.word_id,
            cambridge_senses=cam.senses,
            cambridge_alternatives=cam.alternatives,
            wordnet_synsets=synsets,
        )
