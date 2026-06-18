import json

import pytest

from ai_cost_cutter.compression import (
    collapse_json_whitespace,
    compress,
    dedupe_lines,
    prune_messages,
    remove_filler,
    strip_whitespace,
    truncate_middle,
)
from ai_cost_cutter.tokens import count_tokens


def test_strip_whitespace_collapses_blank_lines_and_trailing():
    text = "a   \n\n\n\nb\t\n   \nc"
    out = strip_whitespace(text)
    assert "\n\n\n" not in out
    assert out.split("\n")[0] == "a"  # trailing spaces removed
    assert out.endswith("c")


def test_strip_whitespace_empty():
    assert strip_whitespace("") == ""


def test_dedupe_lines_removes_repeats_keeps_order():
    text = "alpha\nbeta\nalpha\ngamma\nbeta"
    assert dedupe_lines(text) == "alpha\nbeta\ngamma"


def test_remove_filler_shortens_phrases():
    text = "Please kindly summarize this in order to save time."
    out = remove_filler(text)
    assert "please" not in out.lower()
    assert "kindly" not in out.lower()
    assert "to save time" in out
    assert count_tokens(out) < count_tokens(text)


def test_truncate_middle_respects_budget():
    text = " ".join(f"word{i}" for i in range(500))
    out = truncate_middle(text, max_tokens=50)
    assert count_tokens(out) <= 50
    assert "omitted" in out
    # keeps head and tail
    assert out.startswith("word0")
    assert out.rstrip().endswith("word499")


def test_truncate_middle_noop_when_within_budget():
    text = "short text"
    assert truncate_middle(text, max_tokens=1000) == text


def test_compress_default_is_lossless_tidy():
    text = "Hello   world\n\n\n\nGoodbye   world"
    result = compress(text)
    assert result.tokens_after <= result.tokens_before
    assert "\n\n\n" not in result.compressed


def test_compress_reports_reduction():
    text = "duplicate line\nduplicate line\nduplicate line\nunique line"
    result = compress(text, strategies=["strip_whitespace", "dedupe_lines"])
    assert result.saved_tokens > 0
    assert 0 < result.reduction <= 1
    assert result.ratio == pytest.approx(result.tokens_after / result.tokens_before)


def test_compress_with_max_tokens():
    text = " ".join(["token"] * 400)
    result = compress(text, max_tokens=40)
    assert result.tokens_after <= 40


def test_compress_unknown_strategy_raises():
    with pytest.raises(ValueError):
        compress("x", strategies=["nope"])


def test_compress_accepts_callable_strategy():
    result = compress("HELLO", strategies=[str.lower])
    assert result.compressed == "hello"


def test_collapse_json_whitespace_minifies_object():
    text = 'Here is the data: {\n  "a": 1,\n  "b": [1, 2, 3]\n} thanks'
    out = collapse_json_whitespace(text)
    assert out == 'Here is the data: {"a":1,"b":[1,2,3]} thanks'


def test_collapse_json_whitespace_is_lossless():
    obj = {"name": "Ada", "tags": ["x", "y"], "n": 3, "nested": {"k": [1, 2]}}
    text = "prefix " + json.dumps(obj, indent=4) + " suffix"
    out = collapse_json_whitespace(text)
    # The minified block round-trips to the same object.
    minified = out[len("prefix "): -len(" suffix")]
    assert json.loads(minified) == obj
    assert count_tokens(out) < count_tokens(text)


def test_collapse_json_whitespace_handles_multiple_blocks():
    text = 'a {"x": 1} b [1,  2,   3] c'
    out = collapse_json_whitespace(text)
    assert out == 'a {"x":1} b [1,2,3] c'


def test_collapse_json_whitespace_ignores_braces_in_strings():
    text = '{"note": "use { and } carefully", "v":   2}'
    out = collapse_json_whitespace(text)
    assert json.loads(out) == {"note": "use { and } carefully", "v": 2}
    assert out == '{"note":"use { and } carefully","v":2}'


def test_collapse_json_whitespace_leaves_non_json_untouched():
    # Looks bracket-y but is not valid JSON: must be left exactly as-is.
    text = "if (x) { return y; } // not json"
    assert collapse_json_whitespace(text) == text


def test_collapse_json_whitespace_no_brackets_is_noop():
    text = "plain prose with no structures"
    assert collapse_json_whitespace(text) == text


def test_collapse_json_whitespace_via_registry():
    text = 'log: {"event":  "click",  "count": 7}'
    result = compress(text, strategies=["collapse_json_whitespace"])
    assert '"event":"click"' in result.compressed
    assert result.tokens_after <= result.tokens_before


def test_prune_messages_noop_when_under_budget():
    messages = [{"role": "user", "content": "hi"}]
    result = prune_messages(messages, max_tokens=10_000)
    assert result.dropped == 0
    assert result.messages == messages


def test_prune_messages_drops_oldest_keeps_system_and_recent():
    messages = [
        {"role": "system", "content": "You are a careful assistant."},
        {"role": "user", "content": "old question one " * 20},
        {"role": "assistant", "content": "old answer one " * 20},
        {"role": "user", "content": "old question two " * 20},
        {"role": "assistant", "content": "recent answer " * 5},
        {"role": "user", "content": "the latest question"},
    ]
    full = sum(1 for _ in messages)
    result = prune_messages(messages, max_tokens=120, keep_recent=2)
    roles_kept = [m["role"] for m in result.messages]
    # system retained
    assert result.messages[0]["role"] == "system"
    # last message retained
    assert result.messages[-1]["content"] == "the latest question"
    # something was dropped and we are under budget
    assert result.dropped > 0
    assert result.tokens_after <= 120
    assert len(result.messages) < full


def test_prune_messages_saved_tokens():
    messages = [{"role": "user", "content": "filler " * 100} for _ in range(5)]
    result = prune_messages(messages, max_tokens=80, keep_system=False, keep_recent=1)
    assert result.saved_tokens > 0
