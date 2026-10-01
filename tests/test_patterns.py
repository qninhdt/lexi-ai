import pytest

from lexi_ai.patterns import matches_pattern, validate_pattern
from lexi_ai.text import content_hash, match_key, parse_marked_example, validate_lemma


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


def test_markup_and_exact_text_hash():
    assert parse_marked_example('She <t inf="past">took</t> it off.')[1][0].surface == "took"
    assert content_hash(" a ") != content_hash("a")
    for invalid in ('<t inf="wrong">foo</t>', '<t inf="base">foo', "<t>foo</t>"):
        with pytest.raises(ValueError):
            parse_marked_example(invalid)
