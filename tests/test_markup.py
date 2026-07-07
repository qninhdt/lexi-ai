"""Tests for the example-markup reader (``lexi_ai.markup``).

The ``<t inf="...">`` tag is a deliberate contract: examples are stored WITH the
tags, and this module is the one place that reads them. These tests pin the
parse/strip behavior AND the un-tagged fallback that keeps fake-LLM fixtures
(plain strings) working everywhere downstream.
"""

from lexi_ai.markup import Span, parse_marked_example, strip_markup

# --- tagged input ---------------------------------------------------------


def test_single_tag_parses_surface_and_label():
    clean, spans = parse_marked_example('The snow <t inf="past">glistened</t> in the sun.')
    assert clean == "The snow glistened in the sun."
    assert spans == [Span(surface="glistened", inf="past", start=9, end=18)]
    # The span indexes into the CLEAN text, so a caller can blank it exactly.
    assert clean[spans[0].start : spans[0].end] == "glistened"


def test_multiple_tags_in_one_sentence():
    clean, spans = parse_marked_example(
        'She <t inf="past">brought</t> the children <t inf="base">up</t> alone.'
    )
    assert clean == "She brought the children up alone."
    assert [(s.surface, s.inf) for s in spans] == [("brought", "past"), ("up", "base")]
    for s in spans:
        assert clean[s.start : s.end] == s.surface


def test_multiword_surface():
    clean, spans = parse_marked_example('They <t inf="base">act on behalf of</t> us.')
    assert clean == "They act on behalf of us."
    assert spans[0].surface == "act on behalf of"
    assert clean[spans[0].start : spans[0].end] == "act on behalf of"


# --- un-tagged input (fake-LLM fixtures, robustness) ----------------------


def test_untagged_returns_unchanged_no_spans():
    clean, spans = parse_marked_example("A totally plain example sentence.")
    assert clean == "A totally plain example sentence."
    assert spans == []


def test_strip_markup_display_form():
    assert strip_markup('The wet leaves were <t inf="ing">glistening</t>.') == (
        "The wet leaves were glistening."
    )
    assert strip_markup("No tags here.") == "No tags here."
