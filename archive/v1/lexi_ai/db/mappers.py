"""The single home for ORM to domain to read-model translation.

Callers must eager-load the relationships a builder touches; these functions are pure and
never trigger I/O, so a lazy load here would raise outside greenlet context rather than
quietly working.
"""

from collections.abc import Mapping

from lexi_ai.db.schema import Sense, SenseRelation, Theme, Word
from lexi_ai.models import (
    AliasView,
    Entry,
    FormView,
    LinkView,
    ReferenceView,
    SenseRelationView,
    SenseView,
    ThemeRecord,
    TopicView,
    WordRecord,
)
from lexi_ai.text import render, sense_content_hash
from lexi_ai.vocab import TIER_ORDER

# A themed overlay: {sense_id: (definition, [examples])}.
ThemedOverlay = Mapping[int, tuple[str, list[str]]]


def word_record(word: Word) -> WordRecord:
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
    return ThemeRecord(
        id=theme.id,
        key=theme.theme_key,
        name=theme.name,
        style_prompt=theme.style_prompt,
        description=theme.description,
        tone=theme.tone,
    )


def sense_relation_view(relation: SenseRelation) -> SenseRelationView:
    """Assemble one sense-level relation view, verifying a resolved target.

    The state is derived rather than stored. A resolved edge is additionally
    hash-verified: if the target sense's definition changed since the edge was resolved,
    the edge is surfaced as unresolved and only the target WORD is trusted. That is the
    last safety net for every target-mutation path; it reads as ``pending`` because a
    re-resolve is warranted.
    """
    target = relation.to_sense
    resolved = relation.to_sense_id is not None and target is not None
    if resolved and relation.target_hash != sense_content_hash(target.definition):
        resolved = False
    if resolved:
        state = "resolved"
        to_sense_id = relation.to_sense_id
        to_sense_gloss = target.definition
    else:
        # Unresolvable means an attempt was made and no sense fit; otherwise pending.
        state = "unresolvable" if relation.resolve_attempted_at is not None else "pending"
        if relation.to_sense_id is not None:
            state = "pending"  # a stale-hash demotion warrants another attempt
        to_sense_id = None
        to_sense_gloss = None
    return SenseRelationView(
        rel_type=relation.rel_type,
        to_word_display=render(relation.to_word.norm),
        to_word_id=relation.to_word_id,
        to_word_status=relation.to_word.status,
        to_sense_id=to_sense_id,
        to_sense_gloss=to_sense_gloss,
        wsd_state=state,
    )


def sense_view(sense: Sense, overlay: ThemedOverlay | None = None) -> SenseView:
    """Assemble a sense view, optionally overlaying a theme; the overlay replaces
    the definition and examples only.
    """
    themed = (overlay or {}).get(sense.id)
    return SenseView(
        definition=themed[0] if themed else sense.definition,
        tier=sense.tier,
        pos=sense.pos,
        cefr_level=sense.cefr_level,
        ipa_uk=sense.ipa_uk,
        ipa_us=sense.ipa_us,
        examples=(
            themed[1]
            if themed
            else [example.text for example in sorted(sense.examples, key=_example_order)]
        ),
        references=[
            ReferenceView(source=reference.source, source_ref=reference.source_ref)
            for reference in sense.references
        ],
        forms=[
            FormView(inf=form.inf, surface=form.surface)
            for form in sorted(sense.forms, key=lambda form: form.form_order)
        ],
        guideword=sense.guideword,
        grammar=list(sense.grammar),
        register=sense.register,
        connotation=sense.connotation,
        collocations=[
            collocation.text
            for collocation in sorted(sense.collocations, key=lambda row: row.collocation_order)
        ],
        domain=sense.domain,
        usage_note=sense.usage_note,
        sense_id=sense.id,
        # Never overlaid: a theme restyles definition and examples, not the headword,
        # so this reads from the word on both branches.
        word=render(sense.word.norm),
        word_id=sense.word_id,
        relations=[sense_relation_view(relation) for relation in sense.relations_out],
    )


def entry_view(word: Word, overlay: ThemedOverlay | None = None) -> Entry:
    """Assemble the full entry read model for one word.

    Senses are ordered by tier then by their stored order, so the most central meaning
    leads. Word-level references such as ``word_family`` and ``confused_with`` surface
    through ``links`` via their relation type.
    """
    senses = sorted(
        word.senses, key=lambda sense: (TIER_ORDER.get(sense.tier, 99), sense.sense_order)
    )
    return Entry(
        display=render(word.norm),
        norm=word.norm,
        entry_type=word.entry_type,
        pos=word.pos,
        status=word.status,
        word_id=word.id,
        senses=[sense_view(sense, overlay) for sense in senses],
        aliases=[
            AliasView(
                display=render(alias.alias_norm),
                alias_norm=alias.alias_norm,
                type=alias.type,
                dialect=alias.dialect,
            )
            for alias in word.aliases
        ],
        links=[
            LinkView(
                display=render(link.to_word.norm),
                norm=link.to_word.norm,
                rel_type=link.rel_type,
                word_id=link.to_word.id,
                status=link.to_word.status,
            )
            for link in word.links_out
        ],
        topics=[
            TopicView(name=link.tag.name, title=link.tag.title)
            for link in sorted(word.tags, key=lambda link: link.tag.name)
        ],
    )


def _example_order(example) -> int:  # noqa: ANN001 - an Example row
    return example.example_order
