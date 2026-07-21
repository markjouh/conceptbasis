import pytest

from scripts.dictionary.adjudicate_merge_edges import parse_decisions


def test_parse_decisions_requires_exact_boolean_mapping():
    assert parse_decisions('{"e1":true,"e2":false}', ["e1", "e2"]) == {
        "e1": True,
        "e2": False,
    }
    with pytest.raises(ValueError):
        parse_decisions('{"e1":true}', ["e1", "e2"])
    with pytest.raises(ValueError):
        parse_decisions('{"e1":1,"e2":false}', ["e1", "e2"])
