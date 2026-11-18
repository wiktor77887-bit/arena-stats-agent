"""Builds the DuckDB database from the cleaned CSV and opens connections to it."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

from .names import normalize

DEFAULT_DB_PATH = Path(os.getenv("ARENA_DB", "data/arena.duckdb"))

AUGMENT_COLS = [f"Augment_{i}" for i in range(1, 7)]
ITEM_COLS = [f"Przedmiot_{i}" for i in range(6)]

# Riot internal names that don't match the display name even after normalization.
CHAMPION_OVERRIDES = {
    "MonkeyKing": "Wukong",
    "Nunu": "Nunu i Willump",
    "Renata": "Renata Glasc",
}


def champion_display_names(internal_names, champion_dict: dict[str, str]) -> dict[str, str]:
    """Maps CSV names ("Chogath", "MonkeyKing") to display names ("Cho'Gath", "Wukong")."""
    by_norm = {normalize(v): v for v in champion_dict.values() if v and v != "Brak"}
    mapping = {}
    for name in internal_names:
        if name in CHAMPION_OVERRIDES:
            mapping[name] = CHAMPION_OVERRIDES[name]
        else:
            mapping[name] = by_norm.get(normalize(name), name)
    return mapping


def build_database(
    df: pd.DataFrame,
    champion_dict: dict[str, str],
    augment_dict: dict[str, str],
    item_dict: dict[str, str],
    con: duckdb.DuckDBPyConnection,
) -> dict[str, int]:
    """Creates all tables on the given connection and returns row counts.

    Rows with placement 0 (remakes / API errors) are dropped.
    """
    df = df[df["Miejsce"].between(1, 8)].reset_index(drop=True)

    champ_map = champion_display_names(
        pd.unique(pd.concat([df["Bohater"], df["Sojusznik"]])), champion_dict
    )

    participants = pd.DataFrame(
        {
            "pid": range(len(df)),
            "champion": df["Bohater"].map(champ_map),
            "ally": df["Sojusznik"].map(champ_map),
            "placement": df["Miejsce"].astype("int8"),
        }
    )

    def long_table(cols: list[str], id_name: str) -> pd.DataFrame:
        long = df[cols].copy()
        long["pid"] = participants["pid"]
        long = long.melt(id_vars="pid", var_name="slot", value_name=id_name)
        long = long[long[id_name] != 0]
        long["slot"] = long["slot"].str.extract(r"(\d+)$")[0].astype("int8")
        return long[["pid", "slot", id_name]]

    p_augments = long_table(AUGMENT_COLS, "augment_id")
    p_items = long_table(ITEM_COLS, "item_id")

    champions = pd.DataFrame(
        {"riot_key": list(champ_map.keys()), "name": list(champ_map.values())}
    )
    used_aug = set(p_augments["augment_id"].unique())
    augments = pd.DataFrame(
        [(int(k), v) for k, v in augment_dict.items() if int(k) in used_aug],
        columns=["augment_id", "name"],
    )
    used_items = set(p_items["item_id"].unique())
    items = pd.DataFrame(
        [(int(k), v) for k, v in item_dict.items() if int(k) in used_items],
        columns=["item_id", "name"],
    )

    tables = {
        "participants": participants,
        "participant_augments": p_augments,
        "participant_items": p_items,
        "champions": champions,
        "augments": augments,
        "items": items,
    }
    for name, frame in tables.items():
        con.register("_tmp", frame)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _tmp")
        con.unregister("_tmp")

    con.execute("CREATE INDEX IF NOT EXISTS idx_aug_pid ON participant_augments(pid)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_item_pid ON participant_items(pid)")
    return {name: len(frame) for name, frame in tables.items()}


def connect(path: str | Path = DEFAULT_DB_PATH, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    path = Path(path)
    if read_only and not path.exists():
        raise FileNotFoundError(
            f"Database {path} not found. Run: python scripts/build_db.py"
        )
    return duckdb.connect(str(path), read_only=read_only)
