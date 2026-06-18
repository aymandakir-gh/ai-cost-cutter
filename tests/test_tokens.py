from ai_cost_cutter.tokens import (
    count_messages_tokens,
    count_tokens,
    using_tiktoken,
)


def test_empty_text_is_zero_tokens():
    assert count_tokens("") == 0
    assert count_tokens(None) == 0


def test_token_count_is_positive_and_monotonic():
    short = count_tokens("hello world")
    longer = count_tokens("hello world " * 50)
    assert short >= 1
    assert longer > short


def test_heuristic_never_below_word_count():
    # 10 single-char words: chars/4 would undercount, so word count wins.
    text = " ".join(["a"] * 10)
    assert count_tokens(text) >= 10


def test_count_tokens_is_deterministic():
    text = "The quick brown fox jumps over the lazy dog."
    assert count_tokens(text) == count_tokens(text)


def test_messages_overhead_counts_more_than_raw_content():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]
    combined = "You are helpful." + "Hello!"
    assert count_messages_tokens(messages) > count_tokens(combined)


def test_empty_messages():
    assert count_messages_tokens([]) == 0


def test_using_tiktoken_returns_bool():
    assert isinstance(using_tiktoken(), bool)
