"""Tests for the normalize core (Phase 1).

These are the write==read invariant tests: ``match_key`` is used on BOTH the
write path (indexing generated aliases) and the read path (resolving user
input). If the two disagree, lookups miss forever, so these tests are the
contract for the whole project.
"""

import pytest

from lexi_ai.normalize import PLACEHOLDER_RE, match_key, render

# --- case folding ---------------------------------------------------------


@pytest.mark.parametrize("variant", ["Color", "color", "COLOR", "cOloR"])
def test_case_fold(variant):
    assert match_key(variant) == match_key("color")


# --- diacritics -----------------------------------------------------------


def test_diacritics_cafe():
    assert match_key("café") == match_key("cafe")


def test_diacritics_naive():
    assert match_key("naïve") == match_key("naive")


def test_diacritics_mixed_case_and_accent():
    assert match_key("Résumé") == match_key("resume")


# --- whitespace -----------------------------------------------------------


def test_whitespace_collapses():
    assert match_key("act  on   behalf") == match_key("act on behalf")


def test_whitespace_trimmed():
    assert match_key("  book  ") == match_key("book")


def test_whitespace_tabs_newlines():
    assert match_key("make\tup") == match_key("make up")


# --- placeholder equivalence (the headline requirement) -------------------


def test_placeholder_sb_equivalence():
    assert match_key("act on behalf of {sb}") == match_key("act on behalf of somebody")


def test_placeholder_sb_citation_form():
    # Cambridge notation uses the bare "sb"/"sth" form too.
    assert match_key("act on behalf of {sb}") == match_key("act on behalf of sb")


def test_placeholder_sth_equivalence():
    assert match_key("make {sth} up") == match_key("make something up")


def test_placeholder_possessive_equivalence():
    assert match_key("on {one's} own") == match_key("on your own")
    assert match_key("on {one's} own") == match_key("on one's own")


def test_placeholder_oneself_equivalence():
    assert match_key("look after {oneself}") == match_key("look after yourself")
    assert match_key("look after {oneself}") == match_key("look after oneself")


# --- round-trip property: render reads naturally AND key is stable --------

ROUND_TRIP_SET = [
    ("color", "color"),
    ("act on behalf of {sb}", "act on behalf of somebody"),
    ("make {sth} up", "make something up"),
    ("on {one's} own", "on your own"),
    ("look after {oneself}", "look after yourself"),
]


@pytest.mark.parametrize("norm,expected_display", ROUND_TRIP_SET)
def test_render_reads_naturally(norm, expected_display):
    assert render(norm) == expected_display


@pytest.mark.parametrize("norm,expected_display", ROUND_TRIP_SET)
def test_round_trip_key_stable(norm, expected_display):
    # The core write==read invariant: keying the stored norm and keying its
    # rendered display form must land on the same key.
    assert match_key(norm) == match_key(render(norm))


# --- slash preserved (decision #8) ----------------------------------------


def test_slash_not_split():
    key = match_key("a/c")
    assert "/" in key
    # a/c must NOT equal the two halves joined by a space.
    assert key != match_key("a c")


def test_slash_kept_whole():
    assert match_key("and/or") == "and/or"


# --- real-lemma safety: placeholder regex only matches brace tokens -------


@pytest.mark.parametrize(
    "lemma",
    ["book", "make up", "401(k) plan", "a/c", "cafe", "give up", "someone's"],
)
def test_placeholder_regex_ignores_real_lemmas(lemma):
    assert PLACEHOLDER_RE.search(lemma) is None


@pytest.mark.parametrize("token", ["{sb}", "{sth}", "{one's}", "{oneself}"])
def test_placeholder_regex_matches_tokens(token):
    assert PLACEHOLDER_RE.search(f"do {token} now") is not None


def test_real_word_someones_not_folded():
    # "someone's" contains the substring "one's" but must not fold.
    assert match_key("someone's") != match_key("{one's}")


def test_stones_throw_not_folded():
    # "stone's throw" contains "one's" as a substring — must stay literal.
    # No placeholder sentinel (Private Use Area) may leak into a real lemma.
    key = match_key("a stone's throw")
    assert not any("" <= ch <= "" for ch in key)


def test_match_key_has_no_nul_bytes():
    # Sentinels must be Postgres-storable: NUL (0x00) is rejected by Postgres
    # text columns, so placeholder keys must never contain it (dual-DB, #7).
    for s in [
        "act on behalf of {sb}",
        "make {sth} up",
        "on {one's} own",
        "look after {oneself}",
    ]:
        assert "\x00" not in match_key(s)


# --- render leaves unknown tokens degraded, not crashing -------------------


def test_render_unknown_token_strips_braces():
    assert render("{xyz} thing") == "xyz thing"


def test_render_no_tokens_is_identity():
    assert render("plain phrase") == "plain phrase"


# --- determinism ----------------------------------------------------------


def test_match_key_deterministic():
    s = "Act  On Behalf Of {sb}"
    assert match_key(s) == match_key(s)
