import pytest

from lexi_ai.patterns import matches_pattern, validate_pattern
from lexi_ai.text import (
    canonical_markup,
    content_hash,
    match_key,
    parse_form,
    parse_marked_example,
    strip_markup,
    validate_lemma,
)
from lexi_ai.vocab import INFLECTION_CODES


def test_identity_preserves_lexical_difference():
    assert match_key(" Cafe\u0301  ") == match_key("CAFÉ")
    assert match_key("someone") != match_key("{sb}")
    assert match_key("look up to") != match_key("look up")
    assert match_key("20/20 vision") == "20/20 vision"
    for invalid in ("foo {one}", "foo {wild}", "put (sth) off", "etc."):
        with pytest.raises(ValueError):
            validate_lemma(invalid)


def test_licensed_patterns_are_anchored_and_bounded():
    assert matches_pattern("put up with {sb}", "put up with my old friend")
    assert matches_pattern("take {sth} off", "took her coat off", forms={"take": ["took"]})
    assert not matches_pattern("take {sth} off", "took it")
    assert not matches_pattern("take {sth} off", "took it away")
    assert not matches_pattern("take off", "took it")
    assert matches_pattern("the apple of {one's} eye", "the apple of her eye")
    assert matches_pattern("20/20 vision", "20/20 vision")
    assert not matches_pattern("20/20 vision", "30/20 vision")
    with pytest.raises(ValueError):
        validate_pattern("the {anything} thing")


@pytest.mark.parametrize(
    "pattern,surface,expected",
    [
        ("{sb}", "my old friend", True),
        ("{sth}", "hello-world", True),
        ("{sth}", "go_away", False),
        ("{one's}", "John’s", True),
        ("{oneself}", "myself", True),
        ("{num}", "1.2", True),
        ("the {place}", "the railway station", True),
        ("{doing} {do}", "reading a book go home", True),
        ("have {done}", "have finished the report", True),
        ("have {sth} {done}", "have my car repaired", True),
        ("go {adj}", "go very quiet", True),
        ("do {adv}", "do very well", True),
        ("go {adj}", "go", False),
        ("do {adv}", "do " + " ".join(["well"] * 5), False),
        ("have {done}", "have " + " ".join(["finished"] * 5), False),
        ("{clause}", "we left", True),
        ("{clause}", "left", False),
        ("{clause}", " ".join(["word"] * 13), False),
        ("{sth}{sth}", "gopher", True),
        ("ı {sth}", "i book", True),
    ],
)
def test_slot_languages_and_bounds(pattern, surface, expected):
    assert matches_pattern(pattern, surface) is expected


def test_overlapping_licensed_heads_do_not_hide_longer_matches():
    assert matches_pattern("a{oneself}", "abmyself", forms={"a": ["ab"]})


def test_markup_and_exact_text_hash():
    assert parse_marked_example("She [took|p] it off.")[1][0].surface == "took"
    assert strip_markup("She [took|p] it off.") == "She took it off."
    assert content_hash(" a ") != content_hash("a")
    for invalid in ("[foo|wrong]", "[foo", "[[foo]]"):
        with pytest.raises(ValueError):
            parse_marked_example(invalid)


@pytest.mark.parametrize("code,inflection", INFLECTION_CODES.items())
def test_target_and_form_codes_share_the_same_inflections(code, inflection):
    form = "café" + ("|" + code.upper() if code else "")
    assert parse_form(form) == ("café", inflection)
    marked = "🌍 [" + form + "]!"
    text, spans = parse_marked_example(marked)
    assert text == "🌍 café!"
    assert text[spans[0].start : spans[0].end] == "café"
    assert spans[0].inf == inflection
    assert canonical_markup(marked) == "🌍 [café" + ("|" + code if code else "") + "]!"


@pytest.mark.parametrize("form", ["", "|p", "run|", "run|past", "run|p|p", "[run]", 1])
def test_compact_forms_reject_invalid_surfaces_and_codes(form):
    with pytest.raises(ValueError):
        parse_form(form)


@pytest.mark.parametrize("pattern", ["[have] {done}", "[go|unknown] {adj}"])
def test_patterns_reject_display_tags(pattern):
    with pytest.raises(ValueError):
        validate_pattern(pattern)
