"""Cross-check the SQL tools against an independent pandas computation on the real CSV.

Skipped when data/czyste_dane.csv or data/arena.duckdb is missing (e.g. on CI).
"""

from pathlib import Path

import pandas as pd
import pytest
from pytest import approx

from arena_agent.database import AUGMENT_COLS, ITEM_COLS, connect
from arena_agent.tools import ArenaTools

CSV = Path("data/czyste_dane.csv")
DB = Path("data/arena.duckdb")

pytestmark = pytest.mark.skipif(
    not (CSV.exists() and DB.exists()), reason="real data not available"
)


@pytest.fixture(scope="module")
def df():
    frame = pd.read_csv(CSV)
    return frame[frame["Miejsce"].between(1, 8)]


@pytest.fixture(scope="module")
def tools():
    return ArenaTools(connect(DB))


def summary(places: pd.Series) -> dict:
    return {
        "games": len(places),
        "avg_placement": places.mean(),
        "top2_rate": (places <= 2).mean(),
        "top4_rate": (places <= 4).mean(),
    }


@pytest.mark.parametrize("riot_key, name", [("Jinx", "Jinx"), ("Sett", "Sett"), ("MonkeyKing", "Wukong")])
def test_champion_stats_matches_pandas(df, tools, riot_key, name):
    expected = summary(df.loc[df["Bohater"] == riot_key, "Miejsce"])
    got = tools.champion_stats(name)
    assert got["games"] == expected["games"]
    assert got["avg_placement"] == approx(expected["avg_placement"], abs=0.005)
    assert got["top2_rate"] == approx(expected["top2_rate"], abs=0.0005)
    assert got["top4_rate"] == approx(expected["top4_rate"], abs=0.0005)


def test_top4_rate_matches_wygrana_column(df, tools):
    expected = df.loc[df["Bohater"] == "Jinx", "Wygrana"].mean()
    assert tools.champion_stats("Jinx")["top4_rate"] == approx(expected, abs=0.0005)


def test_top_augment_matches_pandas(df, tools):
    import json

    names = json.load(open("data/slownik_augmentow.json", encoding="utf-8"))
    jinx = df[df["Bohater"] == "Jinx"]
    long = jinx.melt(id_vars="Miejsce", value_vars=AUGMENT_COLS, value_name="aug")
    long = long[long["aug"] != 0]
    long["name"] = long["aug"].astype(str).map(names)
    g = long.groupby("name")["Miejsce"].agg(["count", "mean"])
    g = g[g["count"] >= 100].sort_values(["mean", "count"], ascending=[True, False])

    got = tools.top_augments("Jinx", n=3, min_games=100)["augments"]
    assert [a["augment"] for a in got] == list(g.index[:3])
    for a, (_, row) in zip(got, g.head(3).iterrows()):
        assert a["games"] == row["count"]
        assert a["avg_placement"] == approx(row["mean"], abs=0.005)


def test_compare_items_matches_pandas(df, tools):
    import json

    names = json.load(open("data/slownik_przedmiotow.json", encoding="utf-8"))
    ids_a = {int(k) for k, v in names.items() if v == "Ostrze Nieskończoności"}
    jinx = df[df["Bohater"] == "Jinx"]
    has_a = jinx[ITEM_COLS].isin(ids_a).any(axis=1)
    expected = summary(jinx.loc[has_a, "Miejsce"])

    got = tools.compare_items("Jinx", "Ostrze Nieskończoności", "Huragan Runaana")["item_a"]
    assert got["games"] == expected["games"]
    assert got["avg_placement"] == approx(expected["avg_placement"], abs=0.005)
