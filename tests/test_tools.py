"""Tool tests on the hand-made dataset from conftest.py. No LLM involved."""

import pytest
from pytest import approx

from arena_agent.names import normalize
from arena_agent.tools import TOOL_SPECS


# ------------------------------------------------------------ database build
def test_placement_zero_is_dropped(tools):
    assert tools.con.execute("SELECT count(*) FROM participants").fetchone()[0] == 7


def test_riot_keys_mapped_to_display_names(tools):
    names = {r[0] for r in tools.con.execute("SELECT DISTINCT champion FROM participants").fetchall()}
    assert names == {"Jinx", "Wukong", "Nunu i Willump", "Jhin"}


# ------------------------------------------------------------ champion_stats
def test_champion_stats_numbers(tools):
    s = tools.champion_stats("Jinx")
    # Jinx placements after dropping 0: 1, 2, 5, 8
    assert s["champion"] == "Jinx"
    assert s["games"] == 4
    assert s["avg_placement"] == approx(4.0)
    assert s["first_rate"] == approx(0.25)
    assert s["top2_rate"] == approx(0.5)
    assert s["top4_rate"] == approx(0.5)
    assert s["pick_share"] == approx(4 / 7, abs=1e-4)
    assert s["small_sample"] is True


@pytest.mark.parametrize("spelling", ["jinx", "JINX", " Jinx "])
def test_champion_stats_ignores_case_and_spaces(tools, spelling):
    assert tools.champion_stats(spelling)["games"] == 4


@pytest.mark.parametrize("spelling", ["Wukong", "MonkeyKing", "wukong"])
def test_champion_stats_accepts_riot_key(tools, spelling):
    s = tools.champion_stats(spelling)
    assert s["champion"] == "Wukong"
    assert s["games"] == 1
    assert s["avg_placement"] == approx(3.0)


def test_champion_stats_typo_returns_error_with_suggestion(tools):
    s = tools.champion_stats("jinxx")
    assert "error" in s
    assert s["suggestions"][0] == "Jinx"


def test_champion_stats_unknown_champion(tools):
    s = tools.champion_stats("Pikachu")
    assert "error" in s
    assert "games" not in s


# ------------------------------------------------------------ top_augments
def test_top_augments_ranking_and_numbers(tools):
    r = tools.top_augments("Jinx", n=3, min_games=1)
    # Taniec Ostrzy: places 1, 2, 8 -> 3.67 ; Okrutna Siła: 1, 5 -> 3.0 ; Nietykalność: 5
    assert [a["augment"] for a in r["augments"]] == ["Okrutna Siła", "Taniec Ostrzy", "Nietykalność"]
    first, second, _ = r["augments"]
    assert first["games"] == 2 and first["avg_placement"] == approx(3.0)
    assert second["games"] == 3 and second["avg_placement"] == approx(3.67)
    assert second["top2_rate"] == approx(0.667)
    assert r["baseline"]["avg_placement"] == approx(4.0)
    assert first["delta_vs_baseline"] == approx(-1.0)


def test_top_augments_respects_n_and_min_games(tools):
    r = tools.top_augments("Jinx", n=5, min_games=2)
    assert [a["augment"] for a in r["augments"]] == ["Okrutna Siła", "Taniec Ostrzy"]
    assert len(tools.top_augments("Jinx", n=1, min_games=1)["augments"]) == 1


def test_top_augments_all_champions(tools):
    r = tools.top_augments(None, n=5, min_games=1)
    assert r["champion"] == "ALL"
    by_name = {a["augment"]: a for a in r["augments"]}
    # Nietykalność: Jinx 5, Wukong 3, Nunu 6 -> 4.67
    assert by_name["Nietykalność"]["games"] == 3
    assert by_name["Nietykalność"]["avg_placement"] == approx(4.67)


def test_top_augments_champion_without_augments(tools):
    assert tools.top_augments("Jhin", min_games=1)["augments"] == []


def test_top_augments_unknown_champion(tools):
    assert "error" in tools.top_augments("Pikachu")


# ------------------------------------------------------------ compare_items
def test_compare_items_numbers(tools):
    r = tools.compare_items("Jinx", "Ostrze Nieskończoności", "Kamizelka Kolcza")
    # Ostrze (IDs 200 + 201): places 1, 2 ; Kamizelka: places 2, 5 ; both: place 2
    assert r["item_a"]["games"] == 2
    assert r["item_a"]["avg_placement"] == approx(1.5)
    assert r["item_b"]["games"] == 2
    assert r["item_b"]["avg_placement"] == approx(3.5)
    assert r["both_items"]["games"] == 1
    assert r["avg_placement_diff_a_minus_b"] == approx(-2.0)


def test_compare_items_merges_duplicate_item_ids(tools):
    r = tools.compare_items("Jinx", "ostrze nieskonczonosci", "Buty")
    assert r["item_a"]["name"] == "Ostrze Nieskończoności"
    assert r["item_a"]["games"] == 2


def test_compare_items_item_never_bought(tools):
    r = tools.compare_items("Jhin", "Buty", "Kamizelka Kolcza")
    assert r["item_a"]["games"] == 0
    assert r["avg_placement_diff_a_minus_b"] is None


def test_compare_items_unknown_item(tools):
    r = tools.compare_items("Jinx", "Buty", "Miecz Pikachu")
    assert "error" in r and "item" in r["error"]


# ------------------------------------------------------------ find_name
@pytest.mark.parametrize(
    "text, expected, kind",
    [
        ("jinxx", "Jinx", "champion"),
        ("wukog", "Wukong", "champion"),
        ("nunu", "Nunu i Willump", "champion"),
        ("ostrze niesk", "Ostrze Nieskończoności", "item"),
        ("okrutna sila", "Okrutna Siła", "augment"),
    ],
)
def test_find_name_typos(tools, text, expected, kind):
    best = tools.find_name(text)["matches"][0]
    assert best["name"] == expected
    assert best["kind"] == kind


def test_find_name_exact_flag(tools):
    assert tools.find_name("Jinx")["exact_match"] is True
    assert tools.find_name("jinxx")["exact_match"] is False


def test_find_name_kind_filter(tools):
    matches = tools.find_name("jinx", kind="item")["matches"]
    assert all(m["kind"] == "item" for m in matches)


def test_find_name_nonsense_has_low_score(tools):
    best = tools.find_name("Pikachu")["matches"][0]
    assert best["score"] < 0.6


def test_normalize():
    assert normalize("Kai'Sa") == "kaisa"
    assert normalize("Łuk Śmierci") == "luksmierci"


# ------------------------------------------------------------ registry / dispatch
def test_specs_match_registry(tools):
    assert {s["name"] for s in TOOL_SPECS} == set(tools.registry())


def test_call_dispatches_and_never_raises(tools):
    assert tools.call("champion_stats", {"champion": "Jinx"})["games"] == 4
    assert "error" in tools.call("no_such_tool", {})
    assert "error" in tools.call("champion_stats", {"wrong_arg": 1})
