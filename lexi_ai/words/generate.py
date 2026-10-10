"""Inventory -> parallel batches of up to eight Senses -> atomic publication."""

import asyncio

from sqlalchemy import update

from lexi_ai.db.bulk import insert_identified_rows, insert_rows
from lexi_ai.errors import InvalidHandleError, InvalidOutputError, MissingProviderError
from lexi_ai.inference.prompting import render_prompt
from lexi_ai.references.cambridge import decode_reference_id
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
from lexi_ai.text import match_key, parse_form, validate_lemma
from lexi_ai.vocab import GenerationState, WordRelationType, normalize_pos

from .schemas import (
    EnrichmentBatch,
    InventoryOutput,
    SenseOutput,
    WordOutput,
    validate_evidence,
    validate_inventory,
)
from .storage import consumed_word, insert_contents, publish_identity, target_words


def _headword_field(value, target):
    if not value or value == target:
        return {}
    try:
        if match_key(value) == match_key(target):
            return {}
    except ValueError:
        try:
            if match_key(value.rstrip(".")) == match_key(target):
                return {}
        except ValueError:
            pass
    return {"headword": value}


async def generate_word(
    db, cambridge, llm, reference_id: str, example_count: int, *, target: str, theme_key=None
):
    """Publish one transaction; caller serializes overlapping operations on the same Word."""
    if type(example_count) is not int or example_count < 1:
        raise ValueError("example count must be a positive integer")
    selected_id = decode_reference_id(reference_id)
    target = validate_lemma(target)
    async with db.read() as connection:
        existing = await consumed_word(connection, selected_id, theme_key=theme_key)
        if existing is not None:
            return existing
    entry = await cambridge.fetch_by_id(selected_id)
    if entry is None or not entry.senses:
        raise InvalidHandleError("reference entry has no generation evidence")
    if llm is None:
        raise MissingProviderError("Word generation requires a structured LLM")
    # Prefer the source's canonical slug/headword over a display-form target.
    if target == entry.display and any(s.headword == entry.slug for s in entry.senses):
        target = entry.slug
    elif (
        target == entry.display
        and entry.entry_type == "word"
        and match_key(target) == match_key(entry.slug)
    ):
        headwords = {
            match_key(s.headword): s.headword
            for s in entry.senses
            if s.headword and match_key(s.headword) == match_key(entry.slug)
        }
        if len(headwords) == 1:
            target = next(iter(headwords.values()))
    supporting = await lookup(target)
    cambridge_sources = {f"a{index}": sense for index, sense in enumerate(entry.senses, 1)}
    source_refs = {key: ("cambridge", str(sense.id)) for key, sense in cambridge_sources.items()}
    source_refs.update(
        {f"b{index}": ("wordnet", sense.key) for index, sense in enumerate(supporting, 1)}
    )
    references = [
        {
            "id": key,
            "pos": normalize_pos(sense.pos) or sense.pos or "UNKNOWN",
            "definition": sense.definition,
            **_headword_field(sense.headword, target),
            **({"phrasal": sense.phrase_title} if sense.phrase_title else {}),
        }
        for key, sense in cambridge_sources.items()
    ] + [
        {
            "id": f"b{index}",
            "pos": normalize_pos(sense.pos) or "UNKNOWN",
            "definition": sense.definition,
        }
        for index, sense in enumerate(supporting, 1)
    ]
    grouped = {}
    for reference in references:
        grouped.setdefault(reference.pop("pos"), []).append(reference)
    instruction, data = render_prompt(
        "words/prompts/inventory.jinja",
        target=target,
        references=grouped,
    )
    inventory = await llm.complete(instruction, data, InventoryOutput)
    try:
        inventory = InventoryOutput.model_validate(inventory)
    except ValueError as exc:
        raise InvalidOutputError("invalid Word inventory") from exc
    # A source display may list variants; its stored citation still anchors identity.
    if target == entry.display and match_key(inventory.lemma) == match_key(entry.slug):
        target = inventory.lemma
    validate_inventory(inventory, set(source_refs), target)
    identity = inventory.model_dump(by_alias=True, exclude={"senses"})

    async def enrich(batch):
        instruction, data = render_prompt(
            "words/prompts/enrich_sense.jinja",
            word={key: identity[key] for key in ("lemma", "type", "aliases")},
            senses=[
                dict(sense_id=id, **seed.model_dump(exclude={"references"})) for id, seed in batch
            ],
            examples_per_sense=example_count,
        )
        details = await llm.complete(instruction, data, EnrichmentBatch)
        try:
            details = EnrichmentBatch.model_validate(details)
            by_id = {item.sense_id: item for item in details.senses}
            if set(by_id) != {id for id, _ in batch}:
                raise ValueError("enrichment IDs do not match supplied Senses")
            output = [
                SenseOutput.model_validate(
                    seed.model_dump() | by_id[id].model_dump(by_alias=True, exclude={"sense_id"})
                )
                for id, seed in batch
            ]
        except ValueError as exc:
            raise InvalidOutputError("invalid Sense enrichment") from exc
        return output

    seeds = list(enumerate(inventory.senses, 1))
    tasks = [
        asyncio.create_task(enrich(seeds[offset : offset + 8]))
        for offset in range(0, len(seeds), 8)
    ]
    try:
        batches = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    generated = WordOutput.model_validate(
        identity
        | {
            "senses": [sense.model_dump(by_alias=True) for batch in batches for sense in batch],
        }
    )
    validate_evidence(generated, entry, supporting, target=target)
    sources = {
        key: (order, source) for order, (key, source) in enumerate(cambridge_sources.items())
    }
    async with db.transaction(immediate=True) as session:
        word = await publish_identity(
            session, entry.id, generated.lemma, generated.type, generated.aliases
        )
        sense_rows = []
        for item in generated.senses:
            cited = [sources[ref] for ref in item.references if ref in sources]
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
                    dict(sense_id=id, surface=surface, inf=inf)
                    for id, item in groups
                    for f in item.forms
                    for surface, inf in [parse_form(f)]
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
                    dict(sense_id=id, source=source_refs[ref][0], source_ref=source_refs[ref][1])
                    for id, item in groups
                    for ref in item.references
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
                # Cambridge notation is not a canonical lemma. Do not invent a
                # normalization or a second generation path for ambiguous titles.
                continue
        word_edges = [
            (lemma, kind) for kind, lemmas in generated.related.items() for lemma in lemmas
        ] + [(lemma, WordRelationType.PART_OF_PHRASAL_FAMILY) for lemma in sorted(phrase_lemmas)]
        sense_edges = [
            (id, kind, lemma)
            for id, item in groups
            for kind, lemmas in item.relations.items()
            for lemma in lemmas
        ]
        targets = await target_words(
            session, [lemma for lemma, _ in word_edges] + [lemma for _, _, lemma in sense_edges]
        )
        for table, rows in (
            (
                WordRelation,
                [
                    dict(from_word_id=word.id, to_word_id=target_id, rel_type=kind)
                    for target_id, kind in dict.fromkeys(
                        (targets[match_key(lemma)], kind) for lemma, kind in word_edges
                    )
                    if target_id != word.id
                ],
            ),
            (
                SenseRelation,
                [
                    dict(
                        from_sense_id=sense_id,
                        to_word_id=word_id,
                        rel_type=kind,
                    )
                    for sense_id, kind, word_id in dict.fromkeys(
                        (id, kind, targets[match_key(lemma)])
                        for id, kind, lemma in sense_edges
                    )
                ],
            ),
        ):
            await insert_rows(session, table, rows)
        await session.execute(
            update(Word).where(Word.id == word.id).values(generation_state=GenerationState.DONE)
        )
        identifier = word.id
    if db.content_cache is not None:
        # Publication can also change the displayed identity of relation targets.
        db.content_cache.clear()
    return identifier
