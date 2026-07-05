"""Tests for the LLM generation pipeline (Phase 4).

No live network. The structured LLM is a fake ``Runnable`` returning canned
``GeneratedResult`` objects, so we exercise schema validation, prompt assembly,
the split-vs-alias contract, and the retry wrapper deterministically.
"""

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from lexi_ai.generation.generator import Generator
from lexi_ai.prompts import PromptLoader
from lexi_ai.generation.schemas import (
    GeneratedEntry,
    GeneratedResult,
)
from lexi_ai.references.cambridge import CamSense
from lexi_ai.references.loader import ReferenceBundle
from lexi_ai.references.wordnet import WnSense


def _bundle_book() -> ReferenceBundle:
    return ReferenceBundle(
        word_raw="book",
        entry_type="word",
        cambridge_word_id=11724,
        cambridge_senses=[
            CamSense(
                definition="a written text that can be published",
                guideword="TEXT",
                cefr_level="A1",
                pos="noun",
                phrase_title=None,
                domain=None,
                cambridge_sense_id=1,
                examples=["a good book"],
                synonyms=["volume"],
            )
        ],
        cambridge_synonyms=["volume"],
        wordnet_synsets=[
            WnSense(
                name="book.n.01",
                pos="noun",
                definition="a written work",
                lemmas=["book"],
                examples=["reading a book"],
            )
        ],
    )


def _bundle_expression() -> ReferenceBundle:
    return ReferenceBundle(
        word_raw="act on behalf of",
        entry_type="expression",
        cambridge_word_id=999,
        cambridge_senses=[
            CamSense(
                definition="to represent someone",
                guideword=None,
                cefr_level=None,
                pos="verb",
                phrase_title=None,
                domain=None,
                cambridge_sense_id=42,
            )
        ],
        wordnet_synsets=[],
    )


def _fake_llm(result: GeneratedResult) -> RunnableLambda:
    async def _call(_messages):
        return result

    return RunnableLambda(_call)


# --- prompt assembly ------------------------------------------------------


def _format_bundle(bundle: ReferenceBundle, existing_tags: list = []) -> str:
    alts = (
        ", ".join(f"{a}({t})" for a, t in bundle.cambridge_alternatives)
        if bundle.cambridge_alternatives
        else None
    )
    return PromptLoader.render(
        "senses_generation_user",
        word=bundle.word_raw,
        cambridge_word_raw=bundle.word_raw,
        cambridge_entry_type=bundle.entry_type,
        cambridge_senses=bundle.cambridge_senses,
        cambridge_alternatives=alts,
        wordnet_synsets=bundle.wordnet_synsets,
        existing_tags=existing_tags,
    )


def test_prompt_includes_rubric_and_both_sources():
    body = _format_bundle(_bundle_book())
    assert "CAMBRIDGE" in body
    assert "WORDNET" in body
    assert "book" in body
    # System prompt carries the tier rubric + split rule.
    system_prompt = PromptLoader.render("senses_generation_system")
    assert "core" in system_prompt and "SPLIT" in system_prompt


def test_prompt_marks_empty_wordnet():
    body = _format_bundle(_bundle_expression())
    assert "no synsets" in body.lower()


# --- schema validation ----------------------------------------------------


async def test_generate_returns_valid_result():
    result = GeneratedResult(
        units=[
            GeneratedEntry(
                norm="book",
                entry_type="word",
                pos="noun",
                senses=[{"definition": "a written text", "tier": "core", "cefr_level": "A1"}],
                aliases=[],
            )
        ]
    )
    gen = Generator(structured_llm=_fake_llm(result))
    out = await gen.generate(_bundle_book())
    assert len(out.units) == 1
    assert out.units[0].norm == "book"


def test_invalid_tier_rejected():
    with pytest.raises(ValidationError):
        GeneratedEntry(
            norm="x",
            entry_type="word",
            senses=[{"definition": "d", "tier": "not-a-tier"}],
        )


def test_invalid_alias_type_rejected():
    with pytest.raises(ValidationError):
        GeneratedEntry(
            norm="x",
            entry_type="word",
            senses=[{"definition": "d", "tier": "core"}],
            aliases=[{"alias_norm": "y", "type": "bogus_type"}],
        )


def test_invalid_rel_type_rejected():
    with pytest.raises(ValidationError):
        GeneratedEntry(
            norm="x",
            entry_type="word",
            senses=[{"definition": "d", "tier": "core"}],
            related=[{"norm": "z", "rel_type": "nonsense"}],
        )


def test_entry_requires_at_least_one_sense():
    with pytest.raises(ValidationError):
        GeneratedEntry(norm="x", entry_type="word", senses=[])


# --- split-vs-alias contract ---------------------------------------------


async def test_split_yields_multiple_units():
    # Independent-lemma bundle -> two units.
    result = GeneratedResult(
        units=[
            GeneratedEntry(
                norm="idiom one",
                entry_type="idiom",
                senses=[{"definition": "meaning one", "tier": "common"}],
            ),
            GeneratedEntry(
                norm="idiom two",
                entry_type="idiom",
                senses=[{"definition": "meaning two", "tier": "common"}],
            ),
        ]
    )
    gen = Generator(structured_llm=_fake_llm(result))
    out = await gen.generate(_bundle_book())
    assert len(out.units) == 2


async def test_same_meaning_variant_is_one_unit_with_alias():
    # "log in" / "log on" — same meaning -> ONE unit + alias, not two units.
    result = GeneratedResult(
        units=[
            GeneratedEntry(
                norm="log in",
                entry_type="phrasal_verb",
                senses=[{"definition": "to access a computer system", "tier": "core"}],
                aliases=[{"alias_norm": "log on", "type": "particle"}],
            )
        ]
    )
    gen = Generator(structured_llm=_fake_llm(result))
    out = await gen.generate(_bundle_book())
    assert len(out.units) == 1
    assert len(out.units[0].aliases) == 1
    assert out.units[0].aliases[0].alias_norm == "log on"


async def test_us_canonical_with_uk_alias():
    result = GeneratedResult(
        units=[
            GeneratedEntry(
                norm="color",
                entry_type="word",
                senses=[{"definition": "hue", "tier": "core"}],
                aliases=[{"alias_norm": "colour", "type": "spelling_uk", "dialect": "uk"}],
            )
        ]
    )
    gen = Generator(structured_llm=_fake_llm(result))
    out = await gen.generate(_bundle_book())
    alias = out.units[0].aliases[0]
    assert alias.type == "spelling_uk"
    assert alias.dialect == "uk"


# --- retry wrapper --------------------------------------------------------


async def test_generate_retries_then_succeeds():
    calls = {"n": 0}
    good = GeneratedResult(
        units=[
            GeneratedEntry(
                norm="book",
                entry_type="word",
                senses=[{"definition": "d", "tier": "core"}],
            )
        ]
    )

    async def _flaky(_messages):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")
        return good

    gen = Generator(structured_llm=RunnableLambda(_flaky), base_delay=0.0)
    out = await gen.generate(_bundle_book())
    assert calls["n"] == 2
    assert out.units[0].norm == "book"


async def test_generate_raises_after_exhausting_retries():
    async def _always_fail(_messages):
        raise RuntimeError("permanent")

    gen = Generator(structured_llm=RunnableLambda(_always_fail), max_retries=2, base_delay=0.0)
    with pytest.raises(RuntimeError, match="permanent"):
        await gen.generate(_bundle_book())
