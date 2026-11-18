"""The agent's tools: plain Python functions over the DuckDB database.

Every tool returns a JSON-serializable dict. When a name can't be resolved,
the tool returns {"error": ..., "suggestions": [...]} instead of guessing,
so the model has to call find_name or tell the user the data is missing.
"""

from __future__ import annotations

from typing import Any, Callable

import duckdb

from .names import NameResolver

# Below this many games a result is flagged as a small sample.
SMALL_SAMPLE = 30

# Shared metric columns. Placement is 1-8 (lower is better), top 4 = "win" in Arena.
_METRICS = """
    count(*)                               AS games,
    round(avg(placement), 2)               AS avg_placement,
    round(avg((placement = 1)::INT), 3)    AS first_rate,
    round(avg((placement <= 2)::INT), 3)   AS top2_rate,
    round(avg((placement <= 4)::INT), 3)   AS top4_rate
"""


def _rows(cur: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _one(cur: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    rows = _rows(cur)
    return rows[0] if rows else {}


class ArenaTools:
    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con
        self.resolver = self._build_resolver()

    # ----------------------------------------------------------------- names
    def _build_resolver(self) -> NameResolver:
        champs = self.con.execute("SELECT riot_key, name FROM champions").fetchall()
        champion_pairs = [(n, n) for _, n in champs] + [(k, n) for k, n in champs]
        augments = self.con.execute("SELECT DISTINCT name FROM augments").fetchall()
        items = self.con.execute("SELECT DISTINCT name FROM items").fetchall()
        return NameResolver.from_names(
            {
                "champion": champion_pairs,
                "augment": [(n, n) for (n,) in augments],
                "item": [(n, n) for (n,) in items],
            }
        )

    def _resolve(self, text: str, kind: str) -> tuple[str | None, dict | None]:
        name = self.resolver.exact(text or "", kind)
        if name:
            return name, None
        suggestions = [m.name for m in self.resolver.search(text or "", kind, limit=3) if m.score >= 0.6]
        error = {
            "error": f"Unknown {kind} '{text}'. Not found in the data.",
            "suggestions": suggestions,
            "hint": "Call find_name to resolve typos, or tell the user this "
            f"{kind} is not in the dataset.",
        }
        return None, error

    # ----------------------------------------------------------------- tools
    def champion_stats(self, champion: str) -> dict:
        name, err = self._resolve(champion, "champion")
        if err:
            return err
        stats = _one(
            self.con.execute(
                f"SELECT {_METRICS} FROM participants WHERE champion = ?", [name]
            )
        )
        total = self.con.execute("SELECT count(*) FROM participants").fetchone()[0]
        stats["pick_share"] = round(stats["games"] / total, 4)
        return {"champion": name, **stats, "small_sample": stats["games"] < SMALL_SAMPLE}

    def top_augments(self, champion: str | None = None, n: int = 5, min_games: int = 100) -> dict:
        n = max(1, min(int(n), 20))
        min_games = max(1, int(min_games))
        name = None
        if champion:
            name, err = self._resolve(champion, "champion")
            if err:
                return err

        where = "WHERE p.champion = ?" if name else ""
        params: list[Any] = [name] if name else []
        baseline = _one(
            self.con.execute(
                f"SELECT count(*) AS games, round(avg(placement), 2) AS avg_placement "
                f"FROM participants p {where}",
                params,
            )
        )
        rows = _rows(
            self.con.execute(
                f"""
                SELECT a.name AS augment, {_METRICS}
                FROM participants p
                JOIN participant_augments pa USING (pid)
                JOIN augments a USING (augment_id)
                {where}
                GROUP BY a.name
                HAVING count(*) >= ?
                ORDER BY avg(placement), count(*) DESC
                LIMIT ?
                """,
                params + [min_games, n],
            )
        )
        for r in rows:
            r["delta_vs_baseline"] = round(r["avg_placement"] - baseline["avg_placement"], 2)
        return {
            "champion": name or "ALL",
            "baseline": baseline,
            "min_games": min_games,
            "ranked_by": "avg_placement (lower is better)",
            "augments": rows,
        }

    def compare_items(self, champion: str, item_a: str, item_b: str) -> dict:
        champ, err = self._resolve(champion, "champion")
        if err:
            return err
        a, err = self._resolve(item_a, "item")
        if err:
            return err
        b, err = self._resolve(item_b, "item")
        if err:
            return err

        has = (
            "EXISTS (SELECT 1 FROM participant_items pi JOIN items i USING (item_id) "
            "WHERE pi.pid = p.pid AND i.name = ?)"
        )

        def stats(condition: str, params: list) -> dict:
            s = _one(
                self.con.execute(
                    f"SELECT {_METRICS} FROM participants p WHERE p.champion = ? AND {condition}",
                    [champ] + params,
                )
            )
            s["small_sample"] = s["games"] < SMALL_SAMPLE
            return s

        sa = stats(has, [a])
        sb = stats(has, [b])
        both = stats(f"{has} AND {has}", [a, b])
        diff = None
        if sa["games"] and sb["games"]:
            diff = round(sa["avg_placement"] - sb["avg_placement"], 2)
        return {
            "champion": champ,
            "item_a": {"name": a, **sa},
            "item_b": {"name": b, **sb},
            "both_items": both,
            "avg_placement_diff_a_minus_b": diff,
            "note": "Items are read from the final inventory, so players knocked out "
            "early have fewer items. Compare like with like and treat this as correlation.",
        }

    def find_name(self, text: str, kind: str = "any", limit: int = 3) -> dict:
        kind = kind if kind in ("champion", "augment", "item") else "any"
        limit = max(1, min(int(limit), 10))
        matches = self.resolver.search(text or "", kind, limit)
        return {
            "query": text,
            "kind": kind,
            "matches": [m.as_dict() for m in matches],
            "exact_match": bool(matches) and matches[0].score == 1.0,
        }

    # -------------------------------------------------------- registry
    def registry(self) -> dict[str, Callable[..., dict]]:
        return {
            "champion_stats": self.champion_stats,
            "top_augments": self.top_augments,
            "compare_items": self.compare_items,
            "find_name": self.find_name,
        }

    def call(self, name: str, args: dict | None) -> dict:
        """Single entry point for the agent loop. Never raises: errors go back to the model."""
        fn = self.registry().get(name)
        if fn is None:
            return {"error": f"Unknown tool '{name}'."}
        try:
            return fn(**(args or {}))
        except TypeError as exc:
            return {"error": f"Bad arguments for {name}: {exc}"}


# Provider-neutral tool definitions (JSON Schema). The agent loop converts
# these to the Gemini / Anthropic format.
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "champion_stats",
        "description": (
            "Overall League of Legends Arena stats for one champion: number of games, "
            "average placement (1-8, lower is better), first-place rate, top-2 rate, "
            "top-4 rate and pick share. Requires an exact champion name; use find_name first "
            "if the user's spelling may be off."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "champion": {"type": "string", "description": "Champion name, e.g. 'Jinx'."}
            },
            "required": ["champion"],
        },
    },
    {
        "name": "top_augments",
        "description": (
            "Best augments ranked by average placement (lower is better), for one champion "
            "or across all champions if champion is omitted. Only augments with at least "
            "min_games games are included. Augment names are in Polish."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "champion": {"type": "string", "description": "Champion name. Omit for all champions."},
                "n": {"type": "integer", "description": "How many augments to return (1-20). Default 5."},
                "min_games": {"type": "integer", "description": "Minimum games per augment. Default 100."},
            },
        },
    },
    {
        "name": "compare_items",
        "description": (
            "Compare two items on one champion: games, average placement, top-2 and top-4 "
            "rate for games with item A, with item B, and with both. Item names are in Polish."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "champion": {"type": "string"},
                "item_a": {"type": "string", "description": "Exact item name (Polish)."},
                "item_b": {"type": "string", "description": "Exact item name (Polish)."},
            },
            "required": ["champion", "item_a", "item_b"],
        },
    },
    {
        "name": "find_name",
        "description": (
            "Fuzzy-match a possibly misspelled name to real champion, augment or item names "
            "from the data, e.g. 'jinxx' -> 'Jinx'. Returns candidates with a 0-1 score; "
            "score 1.0 means exact match. A low best score means the name is probably not in the data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["any", "champion", "augment", "item"],
                    "description": "Restrict the search to one kind. Default 'any'.",
                },
                "limit": {"type": "integer", "description": "Max candidates (1-10). Default 3."},
            },
            "required": ["text"],
        },
    },
]
