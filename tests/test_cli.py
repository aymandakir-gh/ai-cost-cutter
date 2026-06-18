import json

import pytest

from ai_cost_cutter.cli import main


def test_estimate_with_explicit_tokens_json(capsys):
    rc = main(
        [
            "estimate",
            "--model",
            "gpt-4o",
            "--input-tokens",
            "1000000",
            "--output-tokens",
            "0",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["model"] == "gpt-4o"
    assert out["total_cost"] == 2.50


def test_estimate_from_prompt_text(capsys):
    rc = main(["estimate", "--model", "gpt-4o-mini", "--prompt", "hello world"])
    assert rc == 0
    assert "total cost" in capsys.readouterr().out


def test_estimate_from_prompt_file(tmp_path, capsys):
    f = tmp_path / "p.txt"
    f.write_text("hello " * 50, encoding="utf-8")
    rc = main(["estimate", "--model", "gpt-4o", "--prompt-file", str(f), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["input_tokens"] > 0


def test_models_listing(capsys):
    rc = main(["models"])
    assert rc == 0
    assert "gpt-4o" in capsys.readouterr().out


def test_models_json(capsys):
    rc = main(["models", "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "gpt-4o" in data
    assert data["gpt-4o"]["provider"] == "openai"


def test_price_compare_table_sorted_cheapest_first(capsys):
    rc = main(
        [
            "price-compare",
            "--input-tokens",
            "1000",
            "--output-tokens",
            "500",
            "-m",
            "gpt-4o",
            "-m",
            "gpt-4o-mini",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if l and not l.startswith("-")]
    # Header then two model rows; cheaper model (mini) listed first.
    body = lines[1:]
    assert body[0].startswith("gpt-4o-mini")
    assert body[1].startswith("gpt-4o")
    assert "x)" in body[1]  # multiplier marker on the pricier model


def test_price_compare_json_totals_over_calls(capsys):
    rc = main(
        [
            "price-compare",
            "--input-tokens",
            "1000000",
            "--output-tokens",
            "0",
            "-m",
            "gpt-4o",
            "--calls",
            "10",
            "--json",
        ]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["calls"] == 10
    entry = data["models"][0]
    assert entry["model"] == "gpt-4o"
    # 1M input tokens at $2.50/1M, 10 calls.
    assert entry["cost_per_call"] == pytest.approx(2.50)
    assert entry["total_cost"] == pytest.approx(25.0)


def test_price_compare_from_prompt_counts_tokens(capsys):
    rc = main(
        ["price-compare", "--prompt", "hello " * 20, "-m", "gpt-4o", "--json"]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["models"][0]["input_tokens"] > 0


def test_price_compare_defaults_to_all_models(capsys):
    rc = main(["price-compare", "--input-tokens", "100", "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    names = {m["model"] for m in data["models"]}
    # A sampling across providers from the default table.
    assert {"gpt-4o", "claude-3-haiku", "gemini-1.5-flash"} <= names


def test_price_compare_unknown_model_errors(capsys):
    rc = main(
        ["price-compare", "--input-tokens", "100", "-m", "no-such-model-xyz"]
    )
    assert rc == 2
    assert "no price" in capsys.readouterr().err


def test_compress_prints_text_and_stats(capsys):
    rc = main(["compress", "--prompt", "dup\ndup\nunique", "--dedupe"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "unique" in captured.out
    assert captured.out.count("dup") == 1  # deduped
    assert "smaller" in captured.err


def test_compress_stats_only_suppresses_text(capsys):
    rc = main(["compress", "--prompt", "hello world", "--stats-only"])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == ""
    assert "tokens" in captured.err


def test_compress_strip_comments_flag(capsys):
    prompt = "```python\nx = 1  # a comment to remove\ny = 2\n```"
    rc = main(["compress", "--prompt", prompt, "--strip-comments"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "a comment to remove" not in captured.out
    assert "x = 1" in captured.out
    assert "smaller" in captured.err


def test_compress_dedupe_near_flag(capsys):
    rc = main(["compress", "--prompt", "Hello World.\nhello world", "--dedupe-near"])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "Hello World."
    assert "smaller" in captured.err


def test_compress_minify_json_flag(capsys):
    rc = main(["compress", "--prompt", 'data {"a":  1,  "b": 2}', "--minify-json"])
    assert rc == 0
    captured = capsys.readouterr()
    assert '{"a":1,"b":2}' in captured.out


def test_dashboard_from_ledger(tmp_path, capsys):
    from ai_cost_cutter.ledger import Ledger

    path = str(tmp_path / "l.jsonl")
    ledger = Ledger(path)
    ledger.record_call("gpt-4o", cost=0.01, baseline_cost=0.05)
    ledger.record_cache_hit("gpt-4o", avoided_cost=0.01)

    rc = main(["dashboard", "--ledger", path, "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["total_calls"] == 2
    assert data["savings"] > 0


def test_dashboard_html_export(tmp_path):
    from ai_cost_cutter.ledger import Ledger

    path = str(tmp_path / "l.jsonl")
    Ledger(path).record_call("gpt-4o", cost=0.01, baseline_cost=0.05)
    out = str(tmp_path / "dash.html")
    rc = main(["dashboard", "--ledger", path, "--html", out])
    assert rc == 0
    with open(out, encoding="utf-8") as fh:
        assert "Cost dashboard" in fh.read()


def test_no_command_prints_help(capsys):
    rc = main([])
    assert rc == 1
    assert "usage" in capsys.readouterr().out.lower()
