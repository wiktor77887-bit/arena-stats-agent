"""Runs the evaluation questions through the agent and reports the scores.

    python scripts/run_eval.py                                   # default Gemini model
    python scripts/run_eval.py --model gemini-3.8-flash --model gemini-3.5-flash-lite
    python scripts/run_eval.py --only champ_basic typo_mundo     # selected cases
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from arena_agent.agent import ArenaAgent  # noqa: E402
from arena_agent.database import DEFAULT_DB_PATH, connect  # noqa: E402
from arena_agent.evaluation import load_cases, run_case, save_report, summarize  # noqa: E402
from arena_agent.gemini import GeminiModel  # noqa: E402
from arena_agent.tools import ArenaTools  # noqa: E402

METRICS = ["pass_rate", "tool_accuracy", "value_accuracy", "grounded_rate", "mentions_rate",
           "errors", "avg_tool_calls", "avg_seconds"]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", help="Gemini model; repeat to compare models.")
    parser.add_argument("--cases", default="eval/questions.json")
    parser.add_argument("--only", nargs="*", help="Run only these case ids.")
    parser.add_argument("--repeat", type=int, default=1, help="Run every case N times (answers vary).")
    parser.add_argument("--delay", type=float, default=1.0, help="Pause between cases (rate limits).")
    parser.add_argument("--out", default="eval/results")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]
    tools = ArenaTools(connect(args.db))
    models = args.model or [None]

    summaries = {}
    for model_name in models:
        model = GeminiModel(model=model_name)
        agent = ArenaAgent(model, tools)
        print(f"\n=== {model.model} ({len(cases)} cases x {args.repeat}) ===")
        results = []
        for run in range(1, args.repeat + 1):
            for case in cases:
                r = run_case(agent, case, tools)
                r.run = run
                results.append(r)
                failed = [k for k, ok in r.checks.items() if not ok]
                status = "PASS" if r.passed else "FAIL " + ",".join(failed)
                print(f"  #{run} {r.id:<22} {status:<28} {len(r.tool_calls)} calls  {r.seconds:5.1f}s")
                for detail in (r.missing_tools, r.missing_values, r.ungrounded):
                    if detail:
                        print(f"      -> {detail}")
                if r.error:
                    print(f"      -> {r.error[:200]}")
                time.sleep(args.delay)
        if args.repeat > 1:
            unstable = sorted({r.id for r in results if not r.passed})
            for case_id in unstable:
                runs = [r for r in results if r.id == case_id]
                print(f"  {case_id}: passed {sum(r.passed for r in runs)}/{len(runs)} runs")
        path = save_report(model.model, results, args.out)
        summaries[model.model] = summarize(results)
        print(f"  report: {path}")

    print("\n" + f"{'metric':<16}" + "".join(f"{m:>24}" for m in summaries))
    for metric in METRICS:
        print(f"{metric:<16}" + "".join(f"{str(s[metric]):>24}" for s in summaries.values()))
    for model_name, s in summaries.items():
        print(f"\n{model_name} by category: {s['by_category']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
