"""Prompts for themed generation (Phase 2).

The formatter emits the numbered neutral sense FACTS (definition, pos, guideword,
tier) but deliberately NOT neutral examples — themed examples are authored fresh
in the theme's voice. Senses are numbered ``1..N`` so the model returns exactly N
themed senses in the same order (index maps to neutral sense id in the api layer).
"""

from collections.abc import Sequence

SYSTEM_PROMPT = """\
You restyle dictionary content into a specific narrative VOICE while keeping every \
meaning identical.

Rules:
- Keep each sense's MEANING exactly as given — restyle only the WORDING of the \
definition into the requested voice. Never change what the word means.
- Author FRESH example sentences in the voice for each sense (do not reuse or \
translate any neutral examples — none are given to you).
- Return exactly one themed sense per numbered neutral sense, IN THE SAME ORDER \
and the SAME COUNT. Do not merge, split, add, or drop senses.
- The style applies to definitions and examples only; do not invent new facts, \
parts of speech, or senses.\
"""


def _sense_facts(
    index: int, definition: str, pos: str | None, guideword: str | None, tier: str
) -> str:
    """One numbered neutral sense block (facts only — NO examples)."""
    lines = [f"Sense {index}:", f"  definition: {definition}", f"  tier: {tier}"]
    if pos:
        lines.append(f"  part of speech: {pos}")
    if guideword:
        lines.append(f"  guideword: {guideword}")
    return "\n".join(lines)


def format_themed(
    style_prompt: str,
    neutral_senses: Sequence[tuple[str, str | None, str | None, str]],
) -> str:
    """Build the user prompt for themed generation.

    ``neutral_senses`` is an ordered list of ``(definition, pos, guideword, tier)``
    — the SAME order the api layer will pass ``sense_ids`` to ``persist_themed``,
    so themed-sense index ``i`` maps to neutral ``sense_ids[i]``.
    """
    blocks = [
        _sense_facts(i + 1, definition, pos, guideword, tier)
        for i, (definition, pos, guideword, tier) in enumerate(neutral_senses)
    ]
    body = "\n".join(blocks)
    return (
        f"VOICE / STYLE:\n{style_prompt}\n\n"
        f"Restyle the following {len(blocks)} sense(s) into that voice. "
        f"Return exactly {len(blocks)} themed sense(s) in the same order.\n\n"
        f"{body}"
    )


THEME_METADATA_SYSTEM_PROMPT = """\
You are a creative dictionary theme designer. Your job is to take a theme identifier key and a basic style prompt, and expand it into a fully-realized dictionary theme profile.

You must output:
1. A creative, catchy display name for the theme (e.g., "The Salty Pirate Captain" for key "pirate").
2. A short creative description/introduction to the style (e.g., "Restyles everything using salty sea jargon, nautical references, and pirate slang.").
3. An expanded, highly detailed, and robust set of instructions (style_prompt) that other LLMs will use to translate dictionary definitions and write example sentences in this specific voice. Be very specific about tone, vocabulary substitutions, and sentence structure.
4. A list of 2-4 tone adjectives representing this style.
"""


def format_theme_metadata(key: str, prompt: str) -> str:
    return (
        f"Theme Key: {key}\n"
        f"Style Prompt Concept: {prompt}\n"
    )
