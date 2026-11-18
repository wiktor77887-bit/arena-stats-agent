"""A tiny hand-made dataset where every expected number can be checked by hand."""

import duckdb
import pandas as pd
import pytest

from arena_agent.database import build_database
from arena_agent.tools import ArenaTools

CHAMPIONS = {"-1": "Brak", "1": "Jinx", "2": "Wukong", "3": "Nunu i Willump", "4": "Jhin"}
AUGMENTS = {"10": "Taniec Ostrzy", "11": "Okrutna Siła", "12": "Nietykalność"}
ITEMS = {
    "100": "Buty",
    "200": "Ostrze Nieskończoności",
    "201": "Ostrze Nieskończoności",  # same item under a second Arena ID
    "300": "Kamizelka Kolcza",
}

#        champion,     ally,        place, augments,    items
ROWS = [
    ("Jinx",       "Nunu",        1, [10, 11],    [200, 100]),
    ("Jinx",       "MonkeyKing",  2, [10],        [201, 300]),
    ("Jinx",       "Nunu",        5, [11, 12],    [300, 100]),
    ("Jinx",       "MonkeyKing",  8, [10],        [100]),
    ("Jinx",       "Nunu",        0, [10],        [200]),      # placement 0 -> dropped
    ("MonkeyKing", "Jinx",        3, [12],        [300]),
    ("Nunu",       "Jinx",        6, [10, 12],    [100, 300]),
    ("Jhin",       "Nunu",        4, [],          []),
]


def make_frame(rows) -> pd.DataFrame:
    records = []
    for champ, ally, place, augs, items in rows:
        rec = {"Bohater": champ, "Sojusznik": ally, "Miejsce": place, "Wygrana": 1 <= place <= 4}
        for i in range(6):
            rec[f"Augment_{i + 1}"] = augs[i] if i < len(augs) else 0
            rec[f"Przedmiot_{i}"] = items[i] if i < len(items) else 0
        records.append(rec)
    return pd.DataFrame(records)


@pytest.fixture()
def tools():
    con = duckdb.connect(":memory:")
    build_database(make_frame(ROWS), CHAMPIONS, AUGMENTS, ITEMS, con)
    yield ArenaTools(con)
    con.close()
