"""Tests for the evaluation checks (no LLM)."""

import json
from pathlib import Path

import pytest

from arena_agent.agent import ArenaAgent, ModelTurn, ToolCall
from arena_agent.evaluation import (
    call_matches,
    check_case,
    extract_numbers,
    number_matches,
    run_case,
    summarize,
    ungrounded_numbers,
    value_in_answer,
)
from tests.test_agent import ScriptedModel


# ------------------------------------------------------------ numbers
@pytest.mark.parametrize(
    "text, value",
    [("4,60", 4.6), ("4.6", 4.6), ("5 518", 5518), ("5 518", 5518), ("117", 117)],
)
def test_extract_numbers_formats(text, value):
    (n,) = extract_numbers(f"wynik: {text} gier")
    assert value in n.values


def test_extract_percent_and_ambiguous_comma():
    (pct,) = extract_numbers("top 4: 65,8%")[1:]
    assert pct.percent and pct.values == (65.8,)
    (n,) = extract_numbers("5,518 games")
    assert set(n.values) == {5.518, 5518}


def test_number_matches_rates_and_rounding():
    (pct,) = extract_numbers("65,8%")
    assert number_matches(pct, [0.658])
    (rounded,) = extract_numbers("66%")
    assert number_matches(rounded, [0.658])
    (wrong,) = extract_numbers("67%")
    assert not number_matches(wrong, [0.658])


def test_ungrounded_finds_made_up_numbers():
    results = [{"avg_placement": 4.6, "top4_rate": 0.485, "games": 5518}]
    answer = "Jinx: 5 518 gier, średnio 4,60 miejsce, top 4 48,5%, winrate 53,2%."
    assert ungrounded_numbers(answer, results) == ["53,2%"]


def test_ungrounded_allows_differences_and_question_numbers():
    results = [{"avg_placement": 4.42}, {"avg_placement": 4.6}]
    assert ungrounded_numbers("Jhin jest lepszy o 0,18 miejsca.", results) == []
    assert ungrounded_numbers("Na patchu 14.5 nie mam danych.", [], "patch 14.5?") == []


def test_mode_baselines_are_allowed():
    assert ungrounded_numbers("Średnia w Arenie to 4,50, a top 4 robi 50% graczy.", []) == []


def test_small_integers_are_ignored():
    assert ungrounded_numbers("1. Jinx\n2. Jhin\ntop 4", []) == []


# ------------------------------------------------------------ tools and values
def test_call_matches_resolves_names(tools):
    exp = {"tool": "champion_stats", "args": {"champion": "Wukong"}}
    assert call_matches(tools, exp, "champion_stats", {"champion": "MonkeyKing"})
    assert not call_matches(tools, exp, "champion_stats", {"champion": "Jinx"})
    assert not call_matches(tools, exp, "top_augments", {"champion": "Wukong"})


def test_call_matches_item_order_and_null_champion(tools):
    exp = {"tool": "compare_items", "args": {"champion": "Jinx", "item_a": "Buty", "item_b": "Kamizelka Kolcza"}}
    assert call_matches(tools, exp, "compare_items", {"champion": "jinx", "item_a": "kamizelka kolcza", "item_b": "buty"})
    glob = {"tool": "top_augments", "args": {"champion": None}}
    assert call_matches(tools, glob, "top_augments", {"n": 5})
    assert not call_matches(tools, glob, "top_augments", {"champion": "Jinx"})


def test_value_in_answer():
    assert value_in_answer("Taniec Ostrzy", "Najlepszy jest **taniec ostrzy**.")
    assert value_in_answer(0.5, "top 4: 50%")
    assert not value_in_answer(4.0, "średnio 4,25")


# ------------------------------------------------------------ whole case
CASE = {
    "id": "c",
    "category": "single_tool",
    "question": "Jak radzi sobie Jinx?",
    "expect_tools": [{"tool": "champion_stats", "args": {"champion": "Jinx"}}],
    "expect_values": [{"tool": "champion_stats", "args": {"champion": "Jinx"}, "fields": ["avg_placement", "top4_rate"]}],
}


def _agent(tools, answer):
    return ArenaAgent(
        ScriptedModel([ModelTurn(None, [ToolCall("champion_stats", {"champion": "jinx"})]), ModelTurn(answer)]),
        tools,
    )


def test_good_answer_passes(tools):
    r = run_case(_agent(tools, "Jinx: 4 gry, średnie miejsce 4,0, top 4 50%."), CASE, tools)
    assert r.passed, r
    assert r.checks == {"tools": True, "values": True, "grounded": True}


def test_hallucinated_number_fails(tools):
    r = run_case(_agent(tools, "Jinx: średnie miejsce 4,0, top 4 50%, winrate 61,3%."), CASE, tools)
    assert not r.passed
    assert r.ungrounded == ["61,3%"]


def test_missing_value_fails(tools):
    r = run_case(_agent(tools, "Jinx ma średnie miejsce 4,0."), CASE, tools)
    assert r.missing_values == ["champion_stats.top4_rate=0.5"]


def test_refusal_case_uses_mentions(tools):
    case = {"id": "o", "category": "out_of_scope", "question": "Runy?", "must_mention_any": ["nie mam"]}
    agent = ArenaAgent(ScriptedModel([ModelTurn("Nie mam danych o runach.")]), tools)
    assert run_case(agent, case, tools).passed


def test_api_error_is_a_failed_case(tools):
    class Broken(ScriptedModel):
        def send_user(self, text):
            raise RuntimeError("503")

    r = run_case(ArenaAgent(Broken([]), tools), CASE, tools)
    assert not r.passed and "503" in r.error
    assert summarize([r])["errors"] == 1


def test_question_file_is_valid(tools):
    cases = json.loads(Path("eval/questions.json").read_text(encoding="utf-8"))
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for c in cases:
        assert c["question"] and c["category"]
        assert c.get("expect_tools") or c.get("must_mention_any")
