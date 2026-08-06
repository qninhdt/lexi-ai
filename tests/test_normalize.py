"""Tests for the normalize core (Phase 1).

These are the write==read invariant tests: ``match_key`` is used on BOTH the
write path (indexing generated aliases) and the read path (resolving user
input). If the two disagree, lookups miss forever, so these tests are the
contract for the whole project.
"""

import unicodedata

import pytest

from lexi_ai.normalize import (
    PLACEHOLDER_RE,
    answer_key,
    fold_diacritics,
    match_key,
    render,
)

# --- case folding ---------------------------------------------------------


@pytest.mark.parametrize("variant", ["Color", "color", "COLOR", "cOloR"])
def test_case_fold(variant):
    assert match_key(variant) == match_key("color")


# --- diacritics -----------------------------------------------------------
#
# match_key PRESERVES diacritics. Folding them here made distinct headwords share
# one key — and `words.match_key` is UNIQUE, so `pâté` could not be stored once
# `pate` existed. Accent-insensitive lookup is kept by registering the folded
# spelling as a `diacritic` alias, which `resolve_key` searches; that is covered
# in the repository tests. `fold_diacritics` is the folding, now separable.


@pytest.mark.parametrize(
    ("accented", "plain"),
    [
        ("résumé", "resume"),
        ("pâté", "pate"),
        ("exposé", "expose"),
        ("rosé", "rose"),
        ("café", "cafe"),
        ("naïve", "naive"),
    ],
)
def test_accented_words_keep_their_own_key(accented, plain):
    """These are different words; one UNIQUE key for both loses an entry."""
    assert match_key(accented) != match_key(plain)


@pytest.mark.parametrize(
    ("accented", "plain"),
    [("résumé", "resume"), ("pâté", "pate"), ("café", "cafe"), ("naïve", "naive")],
)
def test_folding_still_available_for_alias_generation(accented, plain):
    assert fold_diacritics(accented) == plain


def test_folding_leaves_an_unaccented_word_untouched():
    """Callers test `fold_diacritics(x) != x` to decide whether an alias is needed."""
    assert fold_diacritics("resume") == "resume"


def test_diacritic_key_is_still_case_and_whitespace_normalized():
    """Preserving accents must not disable the rest of the pipeline."""
    assert match_key("  Résumé  ") == match_key("résumé")
    assert match_key("CAFÉ") == match_key("café")


# --- Unicode canonicalization ---------------------------------------------
#
# Preserving diacritics must not mean keying on raw code points. `café` composed
# (U+00E9) and `café` decomposed (e + U+0301) render identically and are one
# word; keying them apart produces two indistinguishable dictionary entries,
# which is the collision bug inverted. macOS and several IMEs emit the
# decomposed form, so this is an ordinary input, not an adversarial one.


@pytest.mark.parametrize("word", ["café", "résumé", "naïve", "pâté"])
def test_composed_and_decomposed_spellings_share_one_key(word):
    composed = unicodedata.normalize("NFC", word)
    decomposed = unicodedata.normalize("NFD", word)
    assert composed != decomposed, "the test word must actually differ by encoding"
    assert match_key(composed) == match_key(decomposed)


def test_compatibility_variants_fold_onto_their_plain_letters():
    """A ligature or a fullwidth form is an encoding of the word, not another word."""
    assert match_key("ﬁle") == match_key("file")
    assert match_key("ａbc") == match_key("abc")


def test_canonicalization_does_not_merge_distinct_letters():
    """NFKC must not quietly undo the diacritic fix."""
    assert match_key("résumé") != match_key("resume")


# --- answer_key (grading) --------------------------------------------------
#
# The mirror image of match_key: identity keeps accents apart, comparison folds
# them together. A learner typing `cafe` on a phone keyboard has recalled `café`.


@pytest.mark.parametrize(
    ("typed", "expected"),
    [("cafe", "café"), ("resume", "résumé"), ("naive", "naïve"), ("pate", "pâté")],
)
def test_answer_key_accepts_an_unaccented_answer(typed, expected):
    assert answer_key(typed) == answer_key(expected)


def test_answer_key_inherits_the_match_key_pipeline():
    """Case, whitespace and placeholder folding still apply."""
    assert answer_key("  CAFE  ") == answer_key("cafe")
    assert answer_key("act on behalf of {sb}") == answer_key("act on behalf of somebody")


def test_answer_key_still_separates_genuinely_different_words():
    """Folding accents must not make grading accept anything."""
    assert answer_key("cafe") != answer_key("cage")


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


# --- control/NUL + zero-width folding (core B2/B3, dual-DB) ---------------
#
# match_key is the read AND write key. An embedded NUL crashes the Postgres
# words.match_key INSERT (B2); an invisible zero-width char keys differently
# from the clean form so write-key != read-key and the lookup misses forever
# (B3). Both must be folded, EXCEEDING the sibling keys (which only strip
# _CTRL_RE and would leave the zero-width class in place).


@pytest.mark.parametrize(
    "raw",
    [
        "col\x00or",  # NUL — Postgres text rejects it
        "col\x01or",  # other control char
        "col\x7for",  # DEL
        "col\tor",  # tab (control range, not just \s)
    ],
)
def test_match_key_strips_control_and_nul(raw):
    # The key is Postgres-legal (no NUL/control survives) and deterministic: like
    # the sibling keys, a control char folds to a SPACE (not removed), so a
    # mid-word control becomes a word break — matching tag_key/theme_key. The
    # load-bearing guarantee here is B2 (no NUL reaches words.match_key), proven
    # by the no-NUL assert plus equality to the explicit space-form.
    key = match_key(raw)
    assert "\x00" not in key
    assert all(ord(c) >= 0x20 and ord(c) != 0x7F for c in key)
    assert key == match_key("col or")


@pytest.mark.parametrize(
    "invisible",
    [
        "col\u200bor",  # ZWSP
        "col\ufeffor",  # BOM / zero-width no-break space
        "col\u00ador",  # soft hyphen
        "col\u200dor",  # ZWJ
        "col\u200cor",  # ZWNJ
        "col\u2060or",  # word joiner
    ],
)
def test_match_key_folds_zero_width_format_chars(invisible):
    # An invisible char must vanish (not become a space) so the key equals the
    # clean form — exceeding sibling behavior (_CTRL_RE does NOT cover Cf).
    assert match_key(invisible) == match_key("color")


def test_match_key_zero_width_preserves_placeholder_sentinels():
    # The PUA fold sentinels are category Co, NOT Cf — the format-drop must keep
    # them, so placeholder equivalence still holds when the input carries a ZWSP.
    assert match_key("act on behalf of {sb}\u200b") == match_key("act on behalf of somebody")


def test_match_key_write_read_invariant_for_invisible_input():
    # The headline B3 invariant: a stored norm carrying an invisible char keys
    # identically to the clean user query — otherwise write==read breaks.
    stored = match_key("make\u200b up")  # write path indexes the raw LLM norm
    queried = match_key("make up")  # read path resolves the clean user input
    assert stored == queried


# --- render leaves unknown tokens degraded, not crashing -------------------


def test_render_unknown_token_strips_braces():
    assert render("{xyz} thing") == "xyz thing"


def test_render_no_tokens_is_identity():
    assert render("plain phrase") == "plain phrase"


# --- determinism ----------------------------------------------------------


def test_match_key_deterministic():
    s = "Act  On Behalf Of {sb}"
    assert match_key(s) == match_key(s)
