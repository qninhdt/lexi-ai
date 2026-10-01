"""One selected entry produces one neutral Word, never implicit questions or Sense Linking."""

from sqlalchemy import update

from lexi_ai.db.bulk import insert_identified_rows, insert_rows
from lexi_ai.errors import InvalidHandleError, InvalidOutputError, MissingProviderError
from lexi_ai.inference.prompting import render_prompt
from lexi_ai.references.cambridge import decode_available_id
from lexi_ai.references.wordnet import lookup
from lexi_ai.schema import (
    Collocation,
    Sense,
    SenseForm,
    SensePattern,
    SenseReference,
    SenseRelation,
    Word,
    WordRelation,
)
from lexi_ai.text import match_key, validate_lemma
from lexi_ai.vocab import normalize_pos

from .schemas import WordOutput, validate_evidence
from .storage import consumed_word, insert_contents, publish_identity, target_words


async def generate_word(
    db, cambridge, llm, available_id: str, example_count: int, *, theme_key=None
):
    """Publish one transaction; caller serializes overlapping operations on the same Word."""
    if type(example_count) is not int or example_count < 1:
        raise ValueError("example count must be a positive integer")
    selected_id = decode_available_id(available_id)
    async with db.read() as connection:
        existing = await consumed_word(connection, selected_id, theme_key=theme_key)
        if existing is not None:
            return existing
    entry = await cambridge.fetch_by_id(selected_id)
    if entry is None or not entry.senses:
        raise InvalidHandleError("available entry has no generation evidence")
    if llm is None:
        raise MissingProviderError("Word generation requires a structured LLM")
    supporting = await lookup(entry.slug)
    instruction, data = render_prompt(
        "words/prompts/generate_word.jinja",
        generation_parameters={
            "examples_per_sense": example_count,
        },
        cambridge_entry={
            "display": entry.display,
            "slug": entry.slug,
            "entry_type": entry.entry_type,
            "senses": [
                {
                    "id": f"sense#{sense.id}",
                    "pos": sense.pos,
                    "definition": sense.definition,
                    "examples": sense.examples,
                    "cefr_level": sense.cefr_level,
                    "phrase_title": sense.phrase_title,
                    "ipa_uk": sense.ipa_uk,
                    "ipa_us": sense.ipa_us,
                }
                for sense in entry.senses
            ],
            "alternatives": entry.alternatives,
        },
        wordnet_evidence=[
            {"key": sense.key, "pos": sense.pos, "definition": sense.definition}
            for sense in supporting
        ],
    )
    generated = await llm.complete(instruction, data, WordOutput)
    try:
        generated = WordOutput.model_validate(generated)
    except ValueError as exc:
        raise InvalidOutputError("invalid Word output") from exc
    validate_evidence(generated, entry, supporting)
    if any(len(sense.examples) != example_count for sense in generated.senses):
        raise InvalidOutputError("Word content cardinality differs from the configured counts")
    sources = {str(source.id): (order, source) for order, source in enumerate(entry.senses)}
    async with db.transaction(immediate=True) as session:
        word = await publish_identity(
            session, entry.id, generated.lemma, generated.entry_type, generated.aliases
        )
        sense_rows = []
        for item in generated.senses:
            cited = [
                sources[ref.source_ref.lower().removeprefix("sense#")]
                for ref in item.references
                if ref.source == "cambridge"
            ]
            matching = [
                (order, source) for order, source in cited if normalize_pos(source.pos) == item.pos
            ]
            pronunciation = min(matching, key=lambda pair: pair[0])[1] if matching else None
            sense = dict(
                word_id=word.id,
                pos=item.pos,
                tier=item.tier,
                cefr_level=item.cefr_level,
                register=item.register_,
                usage_note=item.usage_note,
                ipa_uk=pronunciation.ipa_uk if pronunciation else None,
                ipa_us=pronunciation.ipa_us if pronunciation else None,
            )
            sense_rows.append(sense)
        inserted = await insert_identified_rows(session, Sense, sense_rows)
        groups = list(zip(inserted, generated.senses, strict=True))
        await insert_contents(session, groups)
        for table, rows in (
            (
                SenseForm,
                [
                    dict(sense_id=id, surface=f.surface, inf=f.inf)
                    for id, item in groups
                    for f in item.forms
                ],
            ),
            (
                SensePattern,
                [dict(sense_id=id, content=p) for id, item in groups for p in item.patterns],
            ),
            (
                Collocation,
                [dict(sense_id=id, content=c) for id, item in groups for c in item.collocations],
            ),
            (
                SenseReference,
                [
                    dict(sense_id=id, source=r.source, source_ref=r.source_ref)
                    for id, item in groups
                    for r in item.references
                ],
            ),
        ):
            await insert_rows(session, table, rows)
        phrase_lemmas = set()
        for source in entry.senses:
            if not source.phrase_title:
                continue
            try:
                phrase_lemmas.add(validate_lemma(source.phrase_title))
            except ValueError:
                # Reference notation is not a canonical lemma. Do not invent a
                # normalization or a second generation path for ambiguous titles.
                continue
        word_edges = [(item.lemma, item.rel_type) for item in generated.related] + [
            (lemma, "part_of_phrasal_family") for lemma in sorted(phrase_lemmas)
        ]
        sense_edges = [(id, relation) for id, item in groups for relation in item.relations]
        targets = await target_words(
            session, [lemma for lemma, _ in word_edges] + [edge.lemma for _, edge in sense_edges]
        )
        for table, rows in (
            (
                WordRelation,
                [
                    dict(from_word_id=word.id, to_word_id=targets[match_key(lemma)], rel_type=kind)
                    for lemma, kind in word_edges
                    if targets[match_key(lemma)] != word.id
                ],
            ),
            (
                SenseRelation,
                [
                    dict(
                        from_sense_id=id,
                        to_word_id=targets[match_key(edge.lemma)],
                        rel_type=edge.rel_type,
                        gloss=edge.gloss,
                    )
                    for id, edge in sense_edges
                ],
            ),
        ):
            await insert_rows(session, table, rows)
        await session.execute(
            update(Word).where(Word.id == word.id).values(generation_state="done")
        )
        return word.id
