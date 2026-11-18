"""Load the cleaned CSV and the name dictionaries into data/arena.duckdb.

Usage:
    python scripts/build_db.py
    python scripts/build_db.py --csv path/to/czyste_dane.csv --dicts path/to/dir --out data/arena.duckdb
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arena_agent.database import build_database  # noqa: E402


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/czyste_dane.csv"))
    parser.add_argument("--dicts", type=Path, default=Path("data"),
                        help="Folder with slownik_bohaterow/augmentow/przedmiotow.json")
    parser.add_argument("--out", type=Path, default=Path("data/arena.duckdb"))
    args = parser.parse_args()

    start = time.perf_counter()
    df = pd.read_csv(args.csv)
    champions = load_json(args.dicts / "slownik_bohaterow.json")
    augments = load_json(args.dicts / "slownik_augmentow.json")
    items = load_json(args.dicts / "slownik_przedmiotow.json")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    with duckdb.connect(str(args.out)) as con:
        counts = build_database(df, champions, augments, items, con)

    print(f"Built {args.out} in {time.perf_counter() - start:.1f}s")
    for table, n in counts.items():
        print(f"  {table:<22} {n:>10,}")


if __name__ == "__main__":
    main()
