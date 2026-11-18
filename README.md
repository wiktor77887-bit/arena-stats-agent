# Arena Stats Agent

An LLM agent that answers questions about League of Legends **Arena** using real match data
(~900k player-games collected with the Riot API). The model never sees the raw data:
it calls small, tested Python tools that run SQL on DuckDB, and answers only from their results.

> Status: **step 2 of 3 done** – database + tools + tests, Gemini agent loop. Evaluation coming next.

## Data

| table | rows | what |
|---|---:|---|
| `participants` | 899,168 | one row per player per match: champion, ally, placement (1–8) |
| `participant_augments` | 3.2M | augments picked by each player |
| `participant_items` | 5.0M | final inventory of each player |
| `champions` / `augments` / `items` | 172 / 205 / 199 | names (augments and items in Polish) |

Cleaning notes: rows with placement 0 (remakes / API errors, 1,136 rows) are dropped; Riot internal
champion keys (`MonkeyKing`, `Chogath`) are mapped to display names (`Wukong`, `Cho'Gath`); items
that exist under two Arena IDs are merged by name.

## Tools

| tool | returns |
|---|---|
| `champion_stats(champion)` | games, average placement, 1st / top-2 / top-4 rate, pick share |
| `top_augments(champion?, n, min_games)` | best augments by average placement, with delta vs the champion's baseline |
| `compare_items(champion, item_a, item_b)` | stats for games with A, with B and with both |
| `find_name(text, kind?)` | fuzzy match for typos, e.g. `jinxx` → `Jinx` |

Tools never guess: an unknown or misspelled name returns an `error` with suggestions,
so the agent has to resolve it with `find_name` or say the data is missing.

## Agent loop

```
question ─► model ─► tool calls? ─yes─► run tools (ArenaTools.call) ─► JSON results ─► model ─► ...
                          │
                          no ─► final answer
```

`arena_agent/agent.py` holds the provider-neutral loop; `arena_agent/gemini.py` is the Gemini
adapter (google-genai, automatic function calling disabled, so the loop runs every tool itself).
The loop stops after 8 model calls (`max_steps`) and logs each question with its tool calls to
`logs/agent.jsonl`.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

# put czyste_dane.csv and slownik_{bohaterow,augmentow,przedmiotow}.json into data/
python scripts/build_db.py      # ~10 s, creates data/arena.duckdb
pytest                          # 50 tests

# copy .env.example to .env and set GEMINI_API_KEY
python -m arena_agent -v "Jak radzi sobie Jinx?"   # one question, -v shows tool calls
python -m arena_agent                              # interactive chat ('reset', 'exit')
```

`tests/test_tools.py` runs on a tiny hand-made dataset where every number is checked by hand.
`tests/test_real_data.py` recomputes the same stats with pandas straight from the CSV and
checks the SQL tools agree (skipped if the data isn't present).
`tests/test_agent.py` drives the loop with a scripted fake model, so it needs no API key.
