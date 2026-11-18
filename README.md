# Arena Stats Agent

An LLM agent that answers questions about League of Legends **Arena** using real match data
(~900k player-games collected with the Riot API). The model never sees the raw data:
it calls small, tested Python tools that run SQL on DuckDB, and answers only from their results.

> Status: **all 3 steps done** – database + tools + tests, Gemini agent loop, evaluation.

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

## Example

A real run from `logs/agent.jsonl` (the question has a typo: "jinxx"):

```text
> jinxx czy jhin, kto ma lepsze top 4?

tool calls (both in one model turn):
  champion_stats {"champion": "Jinx"} -> games 5518, avg_placement 4.60, top4_rate 0.485
  champion_stats {"champion": "Jhin"} -> games 8516, avg_placement 4.42, top4_rate 0.512

answer:
  Lepszy wskaźnik top 4 ma Jhin.
  * Jhin: top 4 rate 51,2%, średnie miejsce 4,42, liczba gier 8 516
  * Jinx: top 4 rate 48,5%, średnie miejsce 4,60, liczba gier 5 518

model turns: 2
```

Here the model fixed the obvious typo on its own. When it passes a wrong name anyway,
the tool returns an `error` with suggestions and the model has to resolve it before answering.
Every number in the answer appears in the tool results, which is what the `grounded` check verifies.

## Evaluation

`eval/questions.json` holds 17 questions in 6 categories: single tool, augments, items, typos,
comparisons and out-of-scope questions (runes, patches, prices, a champion that doesn't exist).
`arena_agent/evaluation.py` scores each answer with plain code, no LLM judge:

| check | passes when |
|---|---|
| tools | the agent called the expected tools with the right names (after name resolution) |
| values | the answer contains the key facts; expected values come from running a reference tool call on the same database, so nothing is hard-coded |
| grounded | every number in the answer appears in a tool result (or is a difference of two such numbers, a number from the question, or a mode constant like the 4.5 mean placement) |
| mentions | the answer contains a required phrase, e.g. a refusal for out-of-scope questions or a small-sample warning |

```bash
python scripts/run_eval.py --model gemini-3.8-flash --model gemini-3.5-flash-lite --repeat 3
```

Results (3 runs x 17 questions per model, reports in `eval/results/`):

| metric | gemini-3.8-flash | gemini-3.5-flash-lite |
|---|---:|---:|
| pass rate | 51/51 | 51/51 |
| avg tool calls per question | 1.73 | 1.63 |
| avg seconds per question | 8.5 | 1.9 |

Answers vary between runs even at temperature 0, so one run is not enough. In earlier
single runs flash-lite once answered the runes question with no tool calls and made up
Jinx stats (35,481 games, average placement 3.99; the real values are 5,518 and 4.60). The grounding
check caught it. The suite is small, and both models now pass it, so it is a regression test
rather than a benchmark. Harder questions are the next thing to add.

## Limitations

- **Correlation, not causation.** Items come from the final inventory, and players knocked out
  early have fewer items, so `compare_items` partly measures how long a player survived.
  Augment stats have the same problem to a smaller degree.
- **Missing context.** The source CSV has round, HP, map and Guest of Honor columns, but the
  crawler never found those fields in the API, so every row holds the default value. They are not used.
- **One snapshot.** The data covers one period of Arena and is not split by patch, so balance
  changes are mixed together.
- **Small eval suite.** 17 questions, and both models pass all of them. It catches regressions
  and hallucinations like the one above, but it doesn't separate the models. Harder,
  multi-step questions are the next step.
- **Free API tier.** On Gemini's free tier, requests may be used by Google to improve its products.
  Only public game statistics are sent here.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

# put czyste_dane.csv and slownik_{bohaterow,augmentow,przedmiotow}.json into data/
python scripts/build_db.py      # ~10 s, creates data/arena.duckdb
pytest                          # 70 tests

# copy .env.example to .env and set GEMINI_API_KEY
python -m arena_agent -v "Jak radzi sobie Jinx?"   # one question, -v shows tool calls
python -m arena_agent                              # interactive chat ('reset', 'exit')
```

`tests/test_tools.py` runs on a tiny hand-made dataset where every number is checked by hand.
`tests/test_real_data.py` recomputes the same stats with pandas straight from the CSV and
checks the SQL tools agree (skipped if the data isn't present).
`tests/test_agent.py` drives the loop with a scripted fake model, so it needs no API key.
`tests/test_evaluation.py` tests the scoring (number parsing, grounding, tool matching).
