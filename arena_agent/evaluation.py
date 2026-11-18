"""Deterministic evaluation of agent answers. No LLM judge: every check is plain code.

For each test case we check:
  tools     - did the agent call the expected tools with the right (resolved) arguments?
  values    - does the answer contain the key facts? Expected values are not hard-coded:
              they are computed by running a reference tool call on the same database.
  grounded  - does every number in the answer come from a tool result? Numbers the model
              made up (hallucinations) end up in `ungrounded`.
  mentions  - does the answer contain one of the required phrases (e.g. a refusal for
              out-of-scope questions, a warning about a small sample)?
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from .agent import AgentResult, ArenaAgent
from .names import normalize
from .tools import ArenaTools

# Arguments that hold names and must be compared after name resolution.
NAME_ARGS = {"champion": "champion", "item_a": "item", "item_b": "item"}

# Integers this small are mostly list numbering, "top 4", placements 1-8.
IGNORE_SMALL_INTS = 10

# Facts about the game mode itself, not the data: with placements 1-8 the mean placement
# is 4.5 and half of all players finish top 4. Models use them as a baseline.
DOMAIN_CONSTANTS = [4.5, 0.5]

_NUMBER = re.compile(
    r"(?<![\w.,])"
    r"(\d{1,3}(?:[   ]\d{3})+|\d+(?:[.,]\d+)?)"
    r"(?![\w])"
    r"(\s?%)?"
)


# ----------------------------------------------------------------- numbers
@dataclass(frozen=True)
class Number:
    text: str
    values: tuple[float, ...]  # possible readings, e.g. "5,518" = 5.518 or 5518
    decimals: int
    percent: bool


def extract_numbers(text: str) -> list[Number]:
    """Finds numbers written in Polish or English style: 4,60 / 4.6 / 65,8% / 5 518 / 5,518."""
    found = []
    for m in _NUMBER.finditer(text):
        raw, pct = m.group(1), bool(m.group(2))
        if re.fullmatch(r"\d{1,3}(?:[   ]\d{3})+", raw):
            found.append(Number(m.group(0).strip(), (float(re.sub(r"\D", "", raw)),), 0, pct))
            continue
        sep = "," if "," in raw else "." if "." in raw else None
        if sep is None:
            found.append(Number(raw, (float(raw),), 0, pct))
            continue
        whole, frac = raw.split(sep)
        values = [float(f"{whole}.{frac}")]
        if len(frac) == 3 and not pct:  # could be a thousands separator
            values.append(float(whole + frac))
        found.append(Number(m.group(0).strip(), tuple(values), len(frac), pct))
    return found


def _is_trivial(n: Number) -> bool:
    return not n.percent and n.decimals == 0 and n.values[0] <= IGNORE_SMALL_INTS


def collect_numbers(obj: Any) -> list[float]:
    """All numeric values in a nested tool result."""
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in collect_numbers(v)]
    if isinstance(obj, (list, tuple)):
        return [x for v in obj for x in collect_numbers(v)]
    return []


def number_matches(n: Number, pool: list[float]) -> bool:
    """True if the written number is one of the pool values, allowing for rounding and %."""
    tol = 0.5 * 10 ** (-n.decimals) + 1e-9
    for x in n.values:
        for v in pool:
            v = abs(v)
            if n.percent:
                if abs(x - v * 100) <= tol:
                    return True
            elif abs(x - v) <= tol or (v <= 1 and abs(x - v * 100) <= tol):
                return True
    return False


def grounding_pool(results: list[dict]) -> list[float]:
    """Tool numbers plus differences between them (the model may say "0.18 better")."""
    values = sorted({abs(v) for r in results for v in collect_numbers(r)} | set(DOMAIN_CONSTANTS))
    diffs = {round(abs(a - b), 6) for a, b in combinations(values, 2)}
    return values + sorted(diffs)


def ungrounded_numbers(answer: str, results: list[dict], question: str = "") -> list[str]:
    question_numbers = [v for n in extract_numbers(question) for v in n.values]
    pool = grounding_pool(results)
    bad = []
    for n in extract_numbers(answer):
        if _is_trivial(n) or number_matches(n, question_numbers):
            continue
        if not number_matches(n, pool):
            bad.append(n.text)
    return bad


# ----------------------------------------------------------------- tool calls
def _arg_matches(tools: ArenaTools, key: str, expected: Any, actual: Any) -> bool:
    if expected is None:
        return actual in (None, "", "ALL", "all")
    if actual is None:
        return False
    if key in NAME_ARGS:
        return tools.resolver.exact(str(actual), NAME_ARGS[key]) == expected
    if isinstance(expected, (int, float)):
        try:
            return float(actual) == float(expected)
        except (TypeError, ValueError):
            return False
    return normalize(str(actual)) == normalize(str(expected))


def call_matches(tools: ArenaTools, expected: dict, tool: str, args: dict) -> bool:
    if tool != expected["tool"]:
        return False
    exp_args = dict(expected.get("args", {}))
    if tool == "compare_items" and {"item_a", "item_b"} <= exp_args.keys():
        # A vs B and B vs A are the same comparison.
        want = {exp_args.pop("item_a"), exp_args.pop("item_b")}
        got = {tools.resolver.exact(str(args.get(k, "")), "item") for k in ("item_a", "item_b")}
        if want != got:
            return False
    return all(_arg_matches(tools, k, v, args.get(k)) for k, v in exp_args.items())


# ----------------------------------------------------------------- values
def get_path(obj: Any, path: str) -> Any:
    for key in path.split("."):
        obj = obj[int(key)] if isinstance(obj, list) else obj[key]
    return obj


def value_in_answer(value: Any, answer: str) -> bool:
    if isinstance(value, str):
        return normalize(value) in normalize(answer)
    pool = [float(value)]
    return any(number_matches(n, pool) for n in extract_numbers(answer))


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower().replace("ł", "l"))
    return "".join(c for c in text if not unicodedata.combining(c))


# ----------------------------------------------------------------- one case
@dataclass
class CaseResult:
    id: str
    category: str
    question: str
    answer: str = ""
    run: int = 1
    passed: bool = False
    checks: dict[str, bool] = field(default_factory=dict)
    missing_tools: list[dict] = field(default_factory=list)
    missing_values: list[str] = field(default_factory=list)
    ungrounded: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    model_turns: int = 0
    seconds: float = 0.0
    error: str | None = None


def check_case(case: dict, result: AgentResult, tools: ArenaTools) -> CaseResult:
    out = CaseResult(case["id"], case.get("category", ""), case["question"], answer=result.answer)
    out.tool_calls = [{"tool": s.tool, "args": s.args} for s in result.steps]
    out.model_turns = result.model_turns

    if case.get("expect_tools"):
        out.missing_tools = [
            e for e in case["expect_tools"]
            if not any(call_matches(tools, e, s.tool, s.args) for s in result.steps)
        ]
        out.checks["tools"] = not out.missing_tools

    if case.get("expect_values"):
        for spec in case["expect_values"]:
            ref = tools.call(spec["tool"], spec.get("args", {}))
            for path in spec["fields"]:
                value = get_path(ref, path)
                if not value_in_answer(value, result.answer):
                    out.missing_values.append(f"{spec['tool']}.{path}={value}")
        out.checks["values"] = not out.missing_values

    out.ungrounded = ungrounded_numbers(result.answer, [s.result for s in result.steps], case["question"])
    out.checks["grounded"] = not out.ungrounded

    if case.get("must_mention_any"):
        folded = _fold(result.answer)
        out.checks["mentions"] = any(_fold(p) in folded for p in case["must_mention_any"])

    if result.stopped_by_limit:
        out.checks["finished"] = False

    out.passed = all(out.checks.values())
    return out


def run_case(agent: ArenaAgent, case: dict, tools: ArenaTools) -> CaseResult:
    agent.reset()  # each case is a fresh conversation
    start = time.perf_counter()
    try:
        result = agent.ask(case["question"])
    except Exception as exc:  # API errors count as a failed case, the run goes on
        out = CaseResult(case["id"], case.get("category", ""), case["question"], error=repr(exc))
        out.checks = {"error": False}
        out.seconds = round(time.perf_counter() - start, 2)
        return out
    out = check_case(case, result, tools)
    out.seconds = round(time.perf_counter() - start, 2)
    return out


# ----------------------------------------------------------------- summary
def summarize(results: list[CaseResult]) -> dict[str, Any]:
    n = len(results) or 1

    def rate(check: str) -> float | None:
        vals = [r.checks[check] for r in results if check in r.checks]
        return round(sum(vals) / len(vals), 3) if vals else None

    by_cat: dict[str, list[CaseResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)
    return {
        "cases": len(results),
        "pass_rate": round(sum(r.passed for r in results) / n, 3),
        "tool_accuracy": rate("tools"),
        "value_accuracy": rate("values"),
        "grounded_rate": rate("grounded"),
        "mentions_rate": rate("mentions"),
        "errors": sum(r.error is not None for r in results),
        "avg_tool_calls": round(sum(len(r.tool_calls) for r in results) / n, 2),
        "avg_seconds": round(sum(r.seconds for r in results) / n, 2),
        "by_category": {c: f"{sum(r.passed for r in rs)}/{len(rs)}" for c, rs in sorted(by_cat.items())},
    }


def load_cases(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_report(model: str, results: list[CaseResult], directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = directory / f"{model.replace('/', '_')}_{stamp}.json"
    report = {"model": model, "summary": summarize(results), "cases": [asdict(r) for r in results]}
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path
