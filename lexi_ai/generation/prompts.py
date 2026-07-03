"""Prompts for LLM generation (Phase 4).

Contains the system prompt (role, anti-hallucination, canonical/token rules,
US-canonical policy, split-vs-alias rule with concrete examples), the tier
rubric, and a formatter turning a :class:`ReferenceBundle` into a compact user
prompt (Cambridge first, WordNet second, marking when WordNet is empty).
"""

from collections.abc import Sequence

from lexi_ai.references.loader import ReferenceBundle

TIER_RUBRIC = """\
Assign each sense exactly one tier:
- core: foundational, most common, learned first, must-know. The default meaning.
- common: frequent in everyday communication/reading. Learn after core.
- extended: less frequent, context- or domain-specific. Learn once solid.
- rare: rare/archaic/literary/narrow slang/deep jargon. Reference only.\
"""

SPLIT_VS_ALIAS_RULE = """\
SPLIT vs ALIAS (critical):
- Same-meaning surface variants — differ only by preposition/particle/spelling/
  optional element/punctuation — are ONE unit. Record the variants as `aliases`.
  Example: "log in" and "log on" mean the same -> ONE unit, "log on" is an alias.
  Example: US "color" and UK "colour" -> ONE unit, "colour" is a spelling_uk alias.
- Genuinely independent lemmas with DIFFERENT meanings that the source bundled on
  one page -> SEPARATE `units`, one per meaning.
  Example: a page bundling two unrelated idioms -> TWO units.
Do NOT over-split same-meaning variants. Do NOT merge unrelated meanings.\
"""

WORD_REFERENCES_RULE = """\
WORD REFERENCES (emit as `related` items — they are NORMALIZED to real entries,
so give a clean canonical `norm`: US spelling, {sb}/{sth}/{one's}/{oneself}
placeholders, never junk):
- word_family: derivational/inflectional relatives of the headword
  (happy -> happiness, happily, unhappy). Use rel_type="word_family".
- confused_with: commonly-confused near-forms learners mix up
  (affect <-> effect, advice <-> advise). Use rel_type="confused_with".\
"""

SENSE_LABELS_RULE = """\
SENSE LABELS (per-sense enrichments — SYNTHESIZE your own; omit any that don't
clearly apply, an empty field is fine):
- guideword: a 1-3 word sense label disambiguating homographs (bank -> MONEY vs
  RIVER). Use the Cambridge guideword as a hint when present, but author your own.
- grammar: choose 0-3 from EXACTLY this set, only when the sense clearly warrants:
  countable, uncountable, transitive, intransitive, verb + to-infinitive,
  verb + -ing, verb + that-clause, usually singular, usually plural, before noun,
  after verb.
- register: one of {formal, informal, neutral, slang, offensive, humorous, dated,
  literary, technical} — omit when plainly neutral/unmarked.
- connotation: positive, negative, or neutral — the sense's affective tone.
- collocations: 2-6 high-frequency partner phrases showing the word in use
  (make a decision, heavy rain). Illustrative phrases, NOT headwords.\
"""

SYSTEM_PROMPT = f"""\
You are a lexicographer building an English learner's dictionary. You SYNTHESIZE \
the best learner-facing sense list — you do not copy the source verbatim.

Grounding (anti-hallucination):
- Use the provided Cambridge and WordNet anchors as your evidence base.
- Do not invent senses beyond what the anchors plus well-established general \
knowledge support. If an anchor is missing, stay conservative.
- Map each sense back to its sources via `references` (source + source_ref). \
Leaving `references` empty is allowed when nothing anchors it.

Canonical form:
- Write `norm` in canonical form. Represent placeholders as brace tokens: \
{{sb}} (somebody), {{sth}} (something), {{one's}} (possessive), {{oneself}}.
- Prefer the US spelling as `norm`; emit a `spelling_uk` alias pointing back so \
UK input still resolves.
- Resolve "/" and "()" alternatives into explicit, clean alias strings rather \
than leaving slashes or parentheses inside a single string.

{SPLIT_VS_ALIAS_RULE}

{WORD_REFERENCES_RULE}

{TIER_RUBRIC}

cefr_level: when a sense maps to a Cambridge sense that carries a CEFR value, \
reuse that value; otherwise assign one from tier and context.

{SENSE_LABELS_RULE}

TOPICS (broad subject-area tags):
- Assign 1-3 broad topic tags naming the word's subject area (e.g. food, \
business, law, emotion, science, sport), NOT narrow per-sense labels.
- Each topic has a `tag` (lowercase, singular noun, 1-2 words) and a `title` \
(Title Case, human-friendly, e.g. "Business & Finance").
- PREFER reusing a topic from the EXISTING TOPICS list shown below when one \
fits; only invent a new topic when none matches.

Return every field required by the schema. Enums must use the exact allowed \
values.\
"""


def _format_cambridge(bundle: ReferenceBundle) -> str:
    if not bundle.has_cambridge:
        return "CAMBRIDGE: (no entry found)"
    lines = [
        f"CAMBRIDGE entry: {bundle.word_raw}  [entry_type={bundle.entry_type}]",
    ]
    for s in bundle.cambridge_senses:
        head = f"  sense#{s.cambridge_sense_id} (pos={s.pos}, cefr={s.cefr_level}"
        if s.guideword:
            head += f", guideword={s.guideword}"
        head += ")"
        lines.append(head)
        lines.append(f"    def: {s.definition}")
        for ex in s.examples[:3]:
            lines.append(f"    ex: {ex}")
        if s.synonyms:
            lines.append(f"    syn: {', '.join(s.synonyms[:8])}")
        if s.antonyms:
            lines.append(f"    ant: {', '.join(s.antonyms[:8])}")
    if bundle.cambridge_alternatives:
        alts = ", ".join(f"{a}({t})" for a, t in bundle.cambridge_alternatives)
        lines.append(f"  alternatives: {alts}")
    return "\n".join(lines)


def _format_wordnet(bundle: ReferenceBundle) -> str:
    if not bundle.has_wordnet:
        return "WORDNET: (no synsets — this is normal for idioms/expressions)"
    lines = ["WORDNET synsets:"]
    for s in bundle.wordnet_synsets:
        lines.append(f"  {s.name} (pos={s.pos}): {s.definition}")
        if s.lemmas:
            lines.append(f"    lemmas: {', '.join(s.lemmas[:8])}")
        for ex in s.examples[:2]:
            lines.append(f"    ex: {ex}")
    return "\n".join(lines)


def _format_existing_tags(existing_tags: Sequence[tuple[str, str]]) -> str:
    """Render the existing-topic vocab block injected for reuse.

    Each name/title is sanitized at render time (interior newlines/control chars
    collapsed to a space) so a stored value can never break the block format or
    inject instructions into the prompt — defense-in-depth alongside the
    write-path sanitizer.
    """
    if not existing_tags:
        return "EXISTING TOPICS: (none yet — follow the convention)"
    lines = ["EXISTING TOPICS (reuse when one fits; invent new only if none matches):"]
    for name, title in existing_tags:
        clean_name = " ".join(str(name).split())
        clean_title = " ".join(str(title).split())
        lines.append(f"  {clean_name} — {clean_title}")
    return "\n".join(lines)


def format_bundle(bundle: ReferenceBundle, existing_tags: Sequence[tuple[str, str]] = ()) -> str:
    """Render a ReferenceBundle into the user prompt body.

    ``existing_tags`` is the current topic vocabulary (name, title) injected so
    the model reuses topics instead of coining duplicates. Empty by default so
    existing callers/tests are unaffected.
    """
    return (
        f"Generate the dictionary entry for: {bundle.word_raw}\n\n"
        f"{_format_cambridge(bundle)}\n\n"
        f"{_format_wordnet(bundle)}\n\n"
        f"{_format_existing_tags(existing_tags)}"
    )
