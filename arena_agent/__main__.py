"""Command line interface.

    python -m arena_agent "Jak radzi sobie Jinx?"     # one question
    python -m arena_agent                             # interactive chat
    python -m arena_agent -v "..."                    # also print tool calls
"""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from .agent import ArenaAgent, AgentResult, save_log
from .database import DEFAULT_DB_PATH, connect
from .tools import ArenaTools


def print_result(result: AgentResult, verbose: bool) -> None:
    if verbose:
        for i, step in enumerate(result.steps, 1):
            args = json.dumps(step.args, ensure_ascii=False)
            print(f"  [{i}] {step.tool}({args}) -> {json.dumps(step.result, ensure_ascii=False)[:300]}")
        print(f"  ({result.model_turns} model calls, {len(result.steps)} tool calls)\n")
    print(result.answer)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    parser = argparse.ArgumentParser(prog="arena_agent", description="Ask questions about LoL Arena stats.")
    parser.add_argument("question", nargs="*", help="Question; omit for interactive mode.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show tool calls.")
    parser.add_argument("--model", help="Gemini model name (default: GEMINI_MODEL or gemini-3.8-flash).")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--no-log", action="store_true", help="Do not write logs/agent.jsonl.")
    args = parser.parse_args(argv)

    from .gemini import GeminiModel  # imported here so --help works without the SDK

    agent = ArenaAgent(GeminiModel(model=args.model), ArenaTools(connect(args.db)))

    from google.genai import errors

    def handle(question: str) -> None:
        try:
            result = agent.ask(question)
        except errors.APIError as exc:
            print(f"Gemini API error {exc.code}: {exc.message}", file=sys.stderr)
            agent.reset()  # the failed exchange is half-written in the history
            return
        print_result(result, args.verbose)
        if not args.no_log:
            save_log(result)

    if args.question:
        handle(" ".join(args.question))
        return 0

    print("Arena Stats Agent. Type a question, 'reset' for a new conversation, 'exit' to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit", "q"}:
            return 0
        if question.lower() == "reset":
            agent.reset()
            print("(new conversation)")
            continue
        if question:
            handle(question)


if __name__ == "__main__":
    raise SystemExit(main())
