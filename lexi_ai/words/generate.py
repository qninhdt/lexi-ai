"""Inventory -> independent Sense enrichment -> one atomic relational publication."""

import asyncio

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

from .schemas import (
    InventoryOutput,
    SenseEnrichment,
    SenseOutput,
    WordOutput,
    validate_evidence,
    validate_inventory,
)
from .storage import consumed_word, insert_contents, publish_identity, target_words


async def generate_word(
    db, cambridge, llm, available_id: str, example_count: int, *, target: str, theme_key=None
):
    """Publish one transaction; caller serializes overlapping operations on the same Word."""
    if type(example_count) is not int or example_count < 1:
        raise ValueError("example count must be a positive integer")
    selected_id = decode_available_id(available_id)
    target = validate_lemma(target)
    async with db.read() as connection:
        existing = await consumed_word(connection, selected_id, theme_key=theme_key)
        if existing is not None:
            return existing
    entry = await cambridge.fetch_by_id(selected_id)
    if entry is None or not entry.senses:
        raise InvalidHandleError("available entry has no generation evidence")
    if llm is None:
        raise MissingProviderError("Word generation requires a structured LLM")
    supporting = await lookup(target)
    cambridge_sources = {f"c{index}": sense for index, sense in enumerate(entry.senses, 1)}
    source_refs = {key: ("cambridge", str(sense.id)) for key, sense in cambridge_sources.items()}
    source_refs.update(
        {f"w{index}": ("wordnet", sense.key) for index, sense in enumerate(supporting, 1)}
    )
    references = [
        {
            "id": key,
            "pos": sense.pos,
            "definition": sense.definition,
            "cefr_level": sense.cefr_level,
        }
        for key, sense in cambridge_sources.items()
    ] + [
        {"id": f"w{index}", "pos": sense.pos, "definition": sense.definition}
        for index, sense in enumerate(supporting, 1)
    ]
    instruction, data = render_prompt(
        "words/prompts/inventory.jinja",
        target=target,
        references=references,
    )
    inventory = await llm.complete(instruction, data, InventoryOutput)
    try:
        inventory = InventoryOutput.model_validate(inventory)
    except ValueError as exc:
        raise InvalidOutputError("invalid Word inventory") from exc
    validate_inventory(inventory, target)
    identity = inventory.model_dump(by_alias=True, exclude={"senses"})
    semaphore = asyncio.Semaphore(4)

    async def enrich(seed):
        instruction, data = render_prompt(
            "words/prompts/enrich_sense.jinja",
            target=target,
            word=identity,
            sense=seed.model_dump(),
            references=references,
            examples_per_sense=example_count,
        )
        async with semaphore:
            details = await llm.complete(instruction, data, SenseEnrichment)
        try:
            details = SenseEnrichment.model_validate(details)
            output = SenseOutput.model_validate(
                seed.model_dump() | details.model_dump(by_alias=True),
            )
        except ValueError as exc:
            raise InvalidOutputError("invalid Sense enrichment") from exc
        if len(output.examples) != example_count:
            raise InvalidOutputError("Sense example cardinality differs from configured count")
        return output

    tasks = [asyncio.create_task(enrich(seed)) for seed in inventory.senses]
    try:
        senses = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    generated = WordOutput.model_validate(
        identity
        | {
            "senses": [sense.model_dump(by_alias=True) for sense in senses],
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
            cited = [sources[ref] for ref in item.sources if ref in sources]
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
                    dict(sense_id=id, source=source_refs[ref][0], source_ref=source_refs[ref][1])
                    for id, item in groups
                    for ref in item.sources
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
