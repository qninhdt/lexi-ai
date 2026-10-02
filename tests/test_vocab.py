from lexi_ai.vocab import POS_TAGS, normalize_pos


def test_pos_normalization_keeps_known_aliases_and_does_not_guess():
    for pos in POS_TAGS:
        assert normalize_pos(f" {pos.upper()} ") == pos
    assert normalize_pos(" N. ") == "noun"
    assert normalize_pos("adj.") == "adjective"
    assert normalize_pos("modal") == "auxiliary"
    assert normalize_pos("unknown") is None
    assert normalize_pos("") is None
    assert normalize_pos(None) is None
