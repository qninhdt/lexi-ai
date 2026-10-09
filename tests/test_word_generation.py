import asyncio
import json
import re

import pytest
from sqlalchemy import select

from lexi_ai.db.session import Database
from lexi_ai.errors import InvalidHandleError, InvalidOutputError, WordCollisionError
from lexi_ai.references.cambridge import SourceEntry, SourceSense, encode_reference_id
from lexi_ai.references.wordnet import Synset
from lexi_ai.schema import Base, Definition, Sense, SenseReference, Word, WordRelation, WordSource
from lexi_ai.words.generate import generate_word
from lexi_ai.words.schemas import (
    InventoryOutput,
    InventorySense,
    SenseEnrichment,
    WordOutput,
    validate_evidence,
)
from lexi_ai.words.storage import get_word


def payload(lemma="bank", source_ref="a1"):
    return {
        "lemma": lemma,
        "type": "WORD",
        "aliases": [],
        "related": [],
        "senses": [
            {
                "definition": "A place for money",
                "pos": "NOUN",
                "tier": "CORE",
                "cefr_level": "A1",
                "register": None,
                "examples": ['I went to the <t inf="base">bank</t>.'],
                "forms": [],
                "patterns": [],
                "collocations": [],
                "relations": [],
                "references": [source_ref],
            }
        ],
    }


def stage_payload(output, data, schema):
    if schema is InventoryOutput:
        return schema.model_validate(
            {
                **{key: output[key] for key in ("lemma", "type", "aliases", "related")},
                "senses": [
                    {key: sense[key] for key in ("definition", "pos", "references")}
                    for sense in output["senses"]
                ],
            }
        )
    assert schema is SenseEnrichment
    context = json.loads(re.search(r"<sense_request>\s*(.*?)\s*</sense_request>", data).group(1))
    selected = next(
        s
        for s in output["senses"]
        if {key: s[key] for key in ("definition", "pos")} == context["sense"]
    )
    details = {k: v for k, v in selected.items() if k not in {"definition", "pos", "references"}}
    if len(details["examples"]) == 1:
        first = details["examples"][0]
        details["examples"] = [first] + [
            f"Example {i}: {first}" for i in range(1, context["examples_per_sense"])
        ]
    return schema.model_validate(details)


def test_schema_rejects_extras_and_bad_citations():
    entry = SourceEntry(1, "bank", "bank", "word", [SourceSense(101, "noun", "Money")])
    output = WordOutput.model_validate(payload())
    validate_evidence(output, entry, [], target="bank")
    with pytest.raises(InvalidOutputError):
        validate_evidence(
            WordOutput.model_validate(payload(source_ref="a999")), entry, [], target="bank"
        )
    with pytest.raises(ValueError):
        WordOutput.model_validate(payload() | {"units": ["another Word"]})
    with pytest.raises(ValueError):
        WordOutput.model_validate(payload() | {"pos": "noun"})


def test_source_hyphen_is_not_rewritten_as_whitespace():
    entry = SourceEntry(
        1,
        "good-looking",
        "good-looking",
        "word",
        [
            SourceSense(101, "adjective", "attractive"),
        ],
    )
    output = WordOutput.model_validate(payload(lemma="good-looking"))
    output.senses[0].pos = "adjective"
    validate_evidence(output, entry, [], target="good-looking")
    with pytest.raises(InvalidOutputError, match="lemma conflicts"):
        validate_evidence(
            WordOutput.model_validate(payload(lemma="good looking")),
            entry,
            [],
            target="good-looking",
        )
    phrasal = SourceEntry(
        2,
        "look-up",
        "look up",
        "phrasal_verb",
        [
            SourceSense(101, "verb", "find information"),
        ],
    )
    output = WordOutput.model_validate(payload(lemma="look up") | {"type": "PHRASAL_VERB"})
    output.senses[0].pos = "verb"
    validate_evidence(output, phrasal, [], target="look up")


def test_generation_schemas_split_inventory_and_enrichment_with_strict_objects():
    from openai.lib._pydantic import to_strict_json_schema

    schema = to_strict_json_schema(InventoryOutput)
    assert set(schema["$defs"]["EntryType"]["enum"]) == {
        "WORD",
        "PHRASAL_VERB",
        "IDIOM",
        "PHRASE",
        "EXPRESSION",
    }
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["InventorySense"]["additionalProperties"] is False
    assert set(InventorySense.model_fields) == {"definition", "pos", "references"}
    assert set(InventoryOutput.model_fields) == {"lemma", "type", "aliases", "related", "senses"}
    assert schema["properties"]["senses"]["minItems"] == 1
    enrichment = to_strict_json_schema(SenseEnrichment)
    assert enrichment["additionalProperties"] is False
    assert "enum" in enrichment["$defs"]["SenseRelationType"]
    sense_fields = enrichment["properties"]
    assert not {"definition", "pos", "lemma", "senses"} & sense_fields.keys()
    assert not {"sources", "references"} & sense_fields.keys()
    assert not {"id", "references", "ipa_uk", "ipa_us"} & sense_fields.keys()
    assert "entry_type" not in schema["properties"]


class Source:
    def __init__(self, entry):
        self.entry = entry
        self.calls = 0

    async def fetch_by_id(self, entry_id):
        self.calls += 1
        assert entry_id == self.entry.id
        return self.entry


class LLM:
    def __init__(self, data):
        self.data = data
        self.calls = 0

    async def complete(self, instruction, evidence, schema):
        self.calls += 1
        if schema is InventoryOutput:
            assert '"id": "a1"' in evidence
        else:
            assert "references" not in evidence
        return stage_payload(self.data, evidence, schema)


@pytest.fixture
async def db(tmp_path):
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}")
    await database.create_schema(Base.metadata)
    yield database
    await database.close()


@pytest.fixture(autouse=True)
def no_wordnet(monkeypatch):
    async def empty(_citation):
        return []

    monkeypatch.setattr("lexi_ai.words.generate.lookup", empty)


async def test_publish_reuse_and_atomic_rollback(db):
    source = Source(SourceEntry(1, "bank", "bank", "word", [SourceSense(101, "noun", "Money")]))
    llm = LLM(payload())
    handle = encode_reference_id(1)
    first = await generate_word(db, source, llm, handle, 1, target="bank")
    assert await generate_word(db, source, llm, handle, 3, target="bank") == first
    assert llm.calls == 2
    assert source.calls == 1
    async with db.transaction() as session:
        assert (await session.get(Word, first)).generation_state == "DONE"
        assert len((await session.scalars(select(Definition))).all()) == 1
        assert len((await session.scalars(select(Sense))).all()) == 1
        assert len((await session.scalars(select(WordSource))).all()) == 1


async def test_collision_does_not_overwrite_done_word(db):
    first = Source(SourceEntry(1, "bank", "bank", "word", [SourceSense(101, "noun", "Money")]))
    word_id = await generate_word(
        db, first, LLM(payload()), encode_reference_id(1), 1, target="bank"
    )
    second = Source(SourceEntry(2, "bank", "bank", "word", [SourceSense(102, "noun", "Money")]))
    with pytest.raises(WordCollisionError):
        await generate_word(db, second, LLM(payload()), encode_reference_id(2), 1, target="bank")
    async with db.transaction() as session:
        assert (await session.get(Word, word_id)).generation_state == "DONE"
        assert len((await session.scalars(select(WordSource))).all()) == 1


async def test_invalid_output_leaves_no_partial_publish(db):
    source = Source(SourceEntry(1, "bank", "bank", "word", [SourceSense(101, "noun", "Money")]))
    with pytest.raises(InvalidOutputError):
        await generate_word(
            db, source, LLM(payload(source_ref="a999")), encode_reference_id(1), 1, target="bank"
        )
    async with db.transaction() as session:
        assert (await session.scalars(select(Word))).all() == []


async def test_configured_example_count_is_published(db):
    source = Source(SourceEntry(1, "bank", "bank", "word", [SourceSense(101, "noun", "Money")]))
    word_id = await generate_word(
        db, source, LLM(payload()), encode_reference_id(1), 2, target="bank"
    )
    word = await get_word(db, word_id)
    assert len(word.senses[0].examples) == 2


async def test_multi_pos_pronunciation_and_system_derived_phrase_family(db):
    entry = SourceEntry(
        1,
        "bank",
        "bank",
        "word",
        [
            SourceSense(101, "noun", "Money", ipa_uk="bæŋk", ipa_us="bæŋk", phrase_title="bank on"),
            SourceSense(102, "verb", "Turn an aircraft", ipa_uk="verb-uk", ipa_us="verb-us"),
        ],
    )
    output = payload()
    noun = output["senses"][0]
    verb = dict(
        noun,
        pos="VERB",
        references=["a2"],
        definition="Turn an aircraft",
    )
    output["senses"] = [noun, verb]
    word_id = await generate_word(
        db, Source(entry), LLM(output), encode_reference_id(1), 1, target="bank"
    )
    async with db.transaction() as session:
        senses = (
            await session.scalars(select(Sense).where(Sense.word_id == word_id).order_by(Sense.id))
        ).all()
        assert [(s.pos, s.ipa_uk, s.ipa_us) for s in senses] == [
            ("NOUN", "bæŋk", "bæŋk"),
            ("VERB", "verb-uk", "verb-us"),
        ]
        relation = await session.scalar(
            select(WordRelation).where(WordRelation.from_word_id == word_id)
        )
        assert relation.rel_type == "PART_OF_PHRASAL_FAMILY"
        target = await session.get(Word, relation.to_word_id)
        assert (target.lemma, target.generation_state) == ("bank on", "PENDING")


async def test_selected_failures_leave_no_partial_publication(db):
    class MissingSource:
        async def fetch_by_id(self, entry_id):
            return None

    class FailingLLM:
        async def complete(self, *args):
            raise ConnectionError("provider failed")

    with pytest.raises(InvalidHandleError):
        await generate_word(db, MissingSource(), None, "bank", 1, target="bank")
    with pytest.raises(InvalidHandleError):
        await generate_word(db, MissingSource(), None, encode_reference_id(1), 1, target="bank")
    source = Source(SourceEntry(1, "bank", "bank", "word", [SourceSense(101, "noun", "Money")]))
    with pytest.raises(ConnectionError, match="provider failed"):
        await generate_word(db, source, FailingLLM(), encode_reference_id(1), 1, target="bank")
    async with db.transaction() as session:
        assert (await session.scalars(select(Word))).all() == []
        assert (await session.scalars(select(WordSource))).all() == []


async def test_evidence_preserves_lexical_identity_and_local_provenance(db, monkeypatch):
    entry = SourceEntry(
        1,
        "SOURCE_SLUG",
        "bank",
        "word",
        [
            SourceSense(
                987654321,
                "noun",
                "Money",
                ["SOURCE_EXAMPLE"],
                cefr_level="A1",
                ipa_uk="SOURCE_IPA",
                phrase_title="SOURCE_PHRASE",
                headword="SOURCE_HEADWORD",
            ),
            SourceSense(876543210, "verb", "Tilt an aircraft", cefr_level="C1"),
        ],
        alternatives=[("SOURCE_ALTERNATIVE", "SOURCE_ALTERNATIVE_TYPE")],
    )
    synsets = [
        Synset("source.long.raw.key.n.01", "n", "Money storage", ["WORDNET_EXAMPLE"]),
        Synset("source.other.long.key.v.02", "v", "Tilt", []),
    ]

    async def wordnet(citation):
        assert citation == "bank"
        return synsets

    monkeypatch.setattr("lexi_ai.words.generate.lookup", wordnet)
    output = payload()
    output["senses"][0]["references"] = ["a2", "b2", "a1", "b1"]

    class InspectLLM:
        async def complete(self, instruction, data, schema):
            tag = "sense_request" if schema is SenseEnrichment else "word_request"
            request = json.loads(re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", data).group(1))
            common = {
                "target": "bank",
                "references": [
                    {
                        "id": "a1",
                        "pos": "NOUN",
                        "definition": "Money",
                        "cefr_level": "A1",
                        "examples": ["SOURCE_EXAMPLE"],
                        "headword": "SOURCE_HEADWORD",
                        "phrase_title": "SOURCE_PHRASE",
                    },
                    {
                        "id": "a2",
                        "pos": "VERB",
                        "definition": "Tilt an aircraft",
                        "cefr_level": "C1",
                        "examples": [],
                    },
                    {
                        "id": "b1",
                        "synset": "source.long.raw.key.n.01",
                        "examples": ["WORDNET_EXAMPLE"],
                        "pos": "NOUN",
                        "definition": "Money storage",
                    },
                    {
                        "id": "b2",
                        "synset": "source.other.long.key.v.02",
                        "examples": [],
                        "pos": "VERB",
                        "definition": "Tilt",
                    },
                ],
            }
            if schema is InventoryOutput:
                assert request == common
            else:
                assert request == {
                    "word": {k: output[k] for k in ("lemma", "type", "aliases")},
                    "sense": {"definition": "A place for money", "pos": "NOUN"},
                    "examples_per_sense": 1,
                }
            for noise in (
                "SOURCE_SLUG",
                "SOURCE_IPA",
                "SOURCE_ALTERNATIVE",
                "987654321",
                "876543210",
                "generation_parameters",
            ):
                assert noise not in data
            return stage_payload(output, data, schema)

    word_id = await generate_word(
        db, Source(entry), InspectLLM(), encode_reference_id(1), 1, target="bank"
    )
    async with db.read() as connection:
        refs = (
            await connection.execute(select(SenseReference.source, SenseReference.source_ref))
        ).all()
        assert set(refs) == {
            ("cambridge", "987654321"),
            ("cambridge", "876543210"),
            ("wordnet", "source.long.raw.key.n.01"),
            ("wordnet", "source.other.long.key.v.02"),
        }
        assert (await connection.scalar(select(Sense.ipa_uk).where(Sense.word_id == word_id))) == (
            "SOURCE_IPA"
        )


@pytest.mark.parametrize("ref", ["101", "sense#101", "bank.n.01", "C1", "a0", "a01", "a2", "b1"])
def test_citations_accept_only_supplied_local_ids(ref):
    entry = SourceEntry(1, "bank", "bank", "word", [SourceSense(101, "noun", "Money")])
    with pytest.raises(InvalidOutputError, match="not supplied"):
        validate_evidence(
            WordOutput.model_validate(payload(source_ref=ref)), entry, [], target="bank"
        )


def test_duplicate_and_empty_citations_rejected():
    entry = SourceEntry(1, "bank", "bank", "word", [SourceSense(101, "noun", "Money")])
    output = WordOutput.model_validate(payload())
    output.senses[0].references = ["a1", "a1"]
    with pytest.raises(InvalidOutputError, match="duplicate"):
        validate_evidence(output, entry, [], target="bank")
    output.senses[0].references = []
    with pytest.raises(InvalidOutputError, match="no supporting"):
        validate_evidence(output, entry, [], target="bank")


async def test_generated_type_independent_of_source_and_alias_identity(db):
    entry = SourceEntry(1, "colour", "colour", "phrase", [SourceSense(101, "noun", "Hue")])
    output = payload(lemma="color") | {"aliases": ["colour"], "type": "WORD"}
    output["senses"][0]["examples"] = ['It has a bright <t inf="base">color</t>.']
    word_id = await generate_word(
        db, Source(entry), LLM(output), encode_reference_id(1), 1, target="colour"
    )
    word = await get_word(db, word_id)
    assert (word.lemma, word.type, word.aliases) == ("color", "WORD", ["colour"])
    async with db.read() as connection:
        assert (await connection.scalar(select(Word.entry_type))) == "WORD"


@pytest.mark.parametrize("target", ["", " ", None, 1, "bank" * 1000])
async def test_invalid_target_fails_before_source_or_database_io(db, target):
    class UnusedSource:
        async def fetch_by_id(self, entry_id):
            pytest.fail("invalid target reached source")

    with pytest.raises(ValueError):
        await generate_word(db, UnusedSource(), None, encode_reference_id(1), 1, target=target)


def test_source_headword_notation_does_not_block_clean_lemma():
    entry = SourceEntry(
        1,
        "in-charge-of",
        "in charge (of something/someone)",
        "idiom",
        [SourceSense(101, "adjective", "Responsible for something")],
    )
    output = WordOutput.model_validate(payload(lemma="in charge of") | {"type": "PHRASE"})
    validate_evidence(output, entry, [], target="in charge of")


@pytest.mark.parametrize(
    "target,display",
    [
        ("run out of", "run out"),
        ("thank goodness", "thank God, goodness, heaven(s), etc."),
    ],
)
def test_explicit_target_anchors_identity_instead_of_source_display(target, display):
    entry = SourceEntry(1, "source-slug", display, "phrase", [SourceSense(101, "verb", "Meaning")])
    output = WordOutput.model_validate(payload(lemma=target))
    validate_evidence(output, entry, [], target=target)
    with pytest.raises(InvalidOutputError, match="supplied target"):
        validate_evidence(output, entry, [], target="bank")


async def test_generation_passes_explicit_target_to_evidence_validation(db):
    entry = SourceEntry(
        1,
        "run-out",
        "run out",
        "phrasal_verb",
        [SourceSense(101, "verb", "Exhaust")],
    )
    output = payload(lemma="run out of") | {"type": "PHRASAL_VERB"}
    output["senses"][0]["examples"] = ['We <t inf="past">ran out of</t> milk.']
    word_id = await generate_word(
        db,
        Source(entry),
        LLM(output),
        encode_reference_id(1),
        1,
        target="run out of",
    )
    assert (await get_word(db, word_id)).lemma == "run out of"


async def test_usage_note_and_new_metadata_round_trip(db):
    note = 'With a pronoun object, use "put it off", not "put off it".'
    entry = SourceEntry(
        1,
        "put-off",
        "put something off",
        "phrasal_verb",
        [SourceSense(101, "verb", "Postpone", cefr_level="B1")],
    )
    output = payload(lemma="put off") | {"type": "PHRASAL_VERB"}
    sense = output["senses"][0]
    sense.update(
        pos="VERB",
        definition="Postpone",
        tier="LESS_COMMON",
        cefr_level="B1",
        register="INFORMAL",
        usage_note=note,
        examples=['We <t inf="base">put</t> it <t inf="base">off</t>.'],
    )
    word_id = await generate_word(
        db, Source(entry), LLM(output), encode_reference_id(1), 1, target="put off"
    )
    word = await get_word(db, word_id)
    assert (
        word.senses[0].usage_note,
        word.senses[0].tier,
        word.senses[0].cefr_level,
        word.senses[0].register,
    ) == (
        note,
        "LESS_COMMON",
        "B1",
        "INFORMAL",
    )
    output["senses"][0]["cefr_level"] = "B3"
    with pytest.raises(ValueError):
        WordOutput.model_validate(output)


@pytest.mark.parametrize(
    "field,values",
    [
        ("tier", ["CORE", "COMMON", "LESS_COMMON", "RARE"]),
        ("register", [None, "FORMAL", "INFORMAL", "SLANG", "LITERARY", "SPECIALIST"]),
        ("cefr_level", ["A1", "A2", "B1", "B2", "C1", "C2"]),
    ],
)
def test_generation_accepts_only_requested_metadata_vocabulary(field, values):
    for value in values:
        output = payload()
        output["senses"][0][field] = value
        parsed = WordOutput.model_validate(output).senses[0]
        assert getattr(parsed, "register_" if field == "register" else field) == value


@pytest.mark.parametrize(
    "field,value",
    [
        ("tier", "extended"),
        ("tier", None),
        ("tier", "specialist"),
        ("register", "custom-register"),
        ("register", "archaic"),
        ("register", "British"),
        ("register", "humorous"),
        ("register", "figurative"),
        ("cefr_level", None),
        ("cefr_level", "B3"),
    ],
)
def test_generation_rejects_old_or_invalid_metadata(field, value):
    output = payload()
    output["senses"][0][field] = value
    with pytest.raises(ValueError):
        WordOutput.model_validate(output)


@pytest.mark.parametrize("field", ["tier", "register", "cefr_level"])
def test_generation_requires_all_three_metadata_fields_even_nullable_register(field):
    output = payload()
    del output["senses"][0][field]
    with pytest.raises(ValueError):
        WordOutput.model_validate(output)


def test_text_and_native_schemas_share_required_metadata_contract():
    from openai.lib._pydantic import to_strict_json_schema

    for schema in (WordOutput.model_json_schema(), to_strict_json_schema(WordOutput)):
        sense = schema["$defs"]["SenseOutput"]
        properties = sense["properties"]
        assert {"tier", "register", "cefr_level"} <= set(sense["required"])
        assert schema["$defs"]["Tier"]["enum"] == ["CORE", "COMMON", "LESS_COMMON", "RARE"]
        assert schema["$defs"]["CEFRLevel"]["enum"] == ["A1", "A2", "B1", "B2", "C1", "C2"]
        assert "anyOf" not in properties["cefr_level"]
        assert properties["register"]["anyOf"] == [
            {"$ref": "#/$defs/Register"},
            {"type": "null"},
        ]


async def test_enriches_fixed_senses_concurrently_in_inventory_order(db):
    output = payload()
    output["senses"] = [dict(output["senses"][0], definition=f"Meaning {i}") for i in range(6)]

    class ParallelLLM(LLM):
        active = peak = 0
        requests = []

        async def complete(self, instruction, data, schema):
            self.requests.append(schema.__name__)
            if schema is InventoryOutput:
                return await super().complete(instruction, data, schema)
            context = json.loads(re.search(r"<sense_request>\s*(.*?)\s*</sense_request>", data)[1])
            assert set(context) == {"word", "sense", "examples_per_sense"}
            assert "senses" not in context["word"]
            assert set(context["sense"]) == {"definition", "pos"}
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                index = int(context["sense"]["definition"].split()[-1])
                await asyncio.sleep((6 - index) * 0.002)
                details = await super().complete(instruction, data, schema)
                details.examples = [f'Meaning {index} uses <t inf="base">bank</t>.']
                return details
            finally:
                self.active -= 1

    llm = ParallelLLM(output)
    entry = SourceEntry(1, "bank", "bank", "word", [SourceSense(101, "noun", "Money")])
    word_id = await generate_word(db, Source(entry), llm, encode_reference_id(1), 1, target="bank")
    word = await get_word(db, word_id)
    assert llm.requests == ["InventoryOutput"] + ["SenseEnrichment"] * 6
    assert llm.peak == 6 and llm.active == 0
    assert [s.definition.content for s in word.senses] == [f"Meaning {i}" for i in range(6)]
    assert [s.examples[0].content for s in word.senses] == [
        f'Meaning {i} uses <t inf="base">bank</t>.' for i in range(6)
    ]


@pytest.mark.parametrize("defect", ["identity", "references", "duplicate", "blank"])
async def test_bad_inventory_never_starts_enrichment_or_publication(db, defect):
    class BadInventory:
        calls = 0

        async def complete(self, instruction, data, schema):
            self.calls += 1
            assert schema is InventoryOutput
            output = stage_payload(payload(), data, schema).model_dump()
            if defect == "identity":
                output["lemma"] = "river"
            elif defect == "references":
                output["senses"][0]["references"] = ["a999"]
            elif defect == "duplicate":
                output["senses"] *= 2
            else:
                output["senses"][0]["definition"] = "   "
            return output

    llm = BadInventory()
    entry = SourceEntry(1, "bank", "bank", "word", [SourceSense(101, "noun", "Money")])
    with pytest.raises(InvalidOutputError):
        await generate_word(db, Source(entry), llm, encode_reference_id(1), 1, target="bank")
    assert llm.calls == 1
    async with db.read() as connection:
        assert (await connection.execute(select(Word))).all() == []


@pytest.mark.parametrize("field", ["definition", "pos", "references"])
async def test_enrichment_cannot_rewrite_fixed_definition_or_pos(db, field):
    class RewritingLLM(LLM):
        async def complete(self, instruction, data, schema):
            output = await super().complete(instruction, data, schema)
            if schema is SenseEnrichment:
                return output.model_dump(by_alias=True) | {field: "changed"}
            return output

    entry = SourceEntry(1, "bank", "bank", "word", [SourceSense(101, "noun", "Money")])
    with pytest.raises(InvalidOutputError, match="invalid Sense enrichment"):
        await generate_word(
            db,
            Source(entry),
            RewritingLLM(payload()),
            encode_reference_id(1),
            1,
            target="bank",
        )
    async with db.read() as connection:
        assert (await connection.execute(select(Word))).all() == []


@pytest.mark.parametrize("cancel", [False, True])
async def test_failed_or_cancelled_enrichment_drains_children_and_publishes_nothing(db, cancel):
    output = payload()
    output["senses"] = [dict(output["senses"][0], definition=f"Meaning {i}") for i in range(3)]
    started = asyncio.Event()

    class BlockingLLM(LLM):
        active = 0
        stopped = 0

        async def complete(self, instruction, data, schema):
            if schema is InventoryOutput:
                return await super().complete(instruction, data, schema)
            self.active += 1
            if self.active == 3:
                started.set()
            try:
                await started.wait()
                context = json.loads(
                    re.search(
                        r"<sense_request>\s*(.*?)\s*</sense_request>",
                        data,
                    )[1]
                )
                if not cancel and context["sense"]["definition"] == "Meaning 0":
                    raise ConnectionError("enrichment failed")
                await asyncio.Event().wait()
            finally:
                self.active -= 1
                self.stopped += 1

    llm = BlockingLLM(output)
    entry = SourceEntry(1, "bank", "bank", "word", [SourceSense(101, "noun", "Money")])
    task = asyncio.create_task(
        generate_word(
            db,
            Source(entry),
            llm,
            encode_reference_id(1),
            1,
            target="bank",
        )
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else ConnectionError):
        await task
    assert llm.active == 0 and llm.stopped == 3
    async with db.read() as connection:
        assert (await connection.execute(select(Word))).all() == []
        assert (await connection.execute(select(Sense))).all() == []


async def test_affix_pos_is_not_lost_in_inventory_evidence(db):
    entry = SourceEntry(
        1,
        "bank",
        "bank",
        "word",
        [
            SourceSense(101, "prefix", "Bound morpheme", headword="bank-"),
            SourceSense(102, "noun", "Money", headword="bank"),
        ],
    )
    output = payload()
    output["senses"][0]["references"] = ["a2"]

    class InspectLLM:
        async def complete(self, instruction, data, schema):
            tag = "sense_request" if schema is SenseEnrichment else "word_request"
            request = json.loads(re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", data).group(1))
            if schema is InventoryOutput:
                assert request["references"][0]["pos"] == "prefix"
                assert request["references"][0]["headword"] == "bank-"
            else:
                assert "references" not in request
            return stage_payload(output, data, schema)

    await generate_word(db, Source(entry), InspectLLM(), encode_reference_id(1), 1, target="bank")


@pytest.mark.parametrize("field", ["patterns", "collocations", "usage_note", "relations"])
def test_enrichment_target_tags_belong_only_in_examples(field):
    output = payload()
    sense = output["senses"][0]
    tagged = '<t inf="base">bank</t>'
    sense[field] = {
        "patterns": [f"{tagged} on {{sth}}"],
        "collocations": [f"central {tagged}"],
        "usage_note": f"Use {tagged} in this context.",
        "relations": [{"lemma": "institution", "rel_type": "HYPERNYM", "gloss": tagged}],
    }[field]
    with pytest.raises(ValueError, match="plain text"):
        WordOutput.model_validate(output)


@pytest.mark.parametrize("lemma,display", [("a", "A, a"), ("I", "I, i")])
async def test_source_variant_display_accepts_its_stored_citation(db, lemma, display):
    entry = SourceEntry(1, lemma.lower(), display, "word", [SourceSense(101, "pronoun", "Meaning")])
    output = payload(lemma)
    output["senses"][0]["examples"] = [f'<t inf="base">{lemma}</t> appears here.']
    word_id = await generate_word(
        db, Source(entry), LLM(output), encode_reference_id(1), 1, target=display
    )
    assert (await get_word(db, word_id)).lemma == lemma


async def test_variant_display_cannot_accept_an_unrelated_citation(db):
    entry = SourceEntry(1, "a", "A, a", "word", [SourceSense(101, "determiner", "Meaning")])
    output = payload("bank")
    with pytest.raises(InvalidOutputError, match="supplied target"):
        await generate_word(
            db, Source(entry), LLM(output), encode_reference_id(1), 1, target=entry.display
        )


async def test_mixed_source_display_uses_the_headword_confirmed_citation(db):
    entry = SourceEntry(
        1,
        "a",
        "A, a",
        "word",
        [SourceSense(101, "determiner", "One unspecified item", headword="a")],
    )
    output = payload("a")
    output["senses"][0]["examples"] = ['I need <t inf="base">a</t> pen.']

    class InspectLLM(LLM):
        async def complete(self, instruction, data, schema):
            if schema is InventoryOutput:
                context = json.loads(
                    re.search(r"<word_request>\s*(.*?)\s*</word_request>", data, re.S).group(1)
                )
                assert context["target"] == "a"
            return await super().complete(instruction, data, schema)

    word_id = await generate_word(
        db, Source(entry), InspectLLM(output), encode_reference_id(1), 1, target=entry.display
    )
    assert (await get_word(db, word_id)).lemma == "a"
