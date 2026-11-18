"""The agent loop: provider-neutral, so Gemini and other models plug in the same way.

One question goes like this:

    user question ──► model ──► tool calls? ──yes──► run tools ──► results ──► model ──► ...
                                    │
                                    no
                                    ▼
                              final text answer

The model never touches the database. It can only ask for a tool call; the loop runs
the tool (ArenaTools.call), sends the JSON result back, and repeats until the model
answers with plain text or the step limit is hit.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .tools import TOOL_SPECS, ArenaTools

SYSTEM_PROMPT = """\
You are an analyst for League of Legends Arena (2v2v2v2 mode, 8 teams per match).
You answer questions using ONLY the results of the tools you are given. They query real match data.

Rules:
- Never make up numbers, champions, augments or items. Every number in your answer must come from a tool result.
- Placement goes from 1 to 8 and lower is better. Top 4 counts as a win in Arena.
- If a name may be misspelled, call find_name first. If a tool returns an "error", fix the name
  using its suggestions or find_name. If the name still is not found, say it is not in the data.
- Augment and item names in the data are in Polish. Use them exactly as the tools return them.
- If a result has small_sample = true, say the sample is small and the result is unreliable.
- Item stats come from the final inventory, so they show correlation, not causation. Mention this when comparing items.
- If the tools cannot answer the question (e.g. it is about patches, runes or ranked games), say so plainly.
- Answer in the user's language. Be concise: give the key numbers (games, average placement, top-4 rate).
"""

DEFAULT_MAX_STEPS = 8


# ----------------------------------------------------------------- data types
@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    id: str | None = None


@dataclass
class ModelTurn:
    """What the model produced in one step: some text, some tool calls, or both."""

    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class Step:
    """One executed tool call, kept for logging and evaluation."""

    tool: str
    args: dict[str, Any]
    result: dict[str, Any]
    seconds: float


@dataclass
class AgentResult:
    question: str
    answer: str
    steps: list[Step]
    model_turns: int
    stopped_by_limit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChatModel(Protocol):
    """Adapter around one LLM provider. It keeps the conversation in its own native format."""

    def reset(self, system_prompt: str, tool_specs: list[dict[str, Any]]) -> None: ...

    def send_user(self, text: str) -> ModelTurn: ...

    def send_tool_results(self, results: list[tuple[ToolCall, dict[str, Any]]]) -> ModelTurn: ...


# ----------------------------------------------------------------- the loop
class ArenaAgent:
    def __init__(
        self,
        model: ChatModel,
        tools: ArenaTools,
        max_steps: int = DEFAULT_MAX_STEPS,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.model = model
        self.tools = tools
        self.max_steps = max_steps
        self.system_prompt = system_prompt
        self.reset()

    def reset(self) -> None:
        """Start a new conversation (forget earlier questions)."""
        self.model.reset(self.system_prompt, TOOL_SPECS)

    def ask(self, question: str) -> AgentResult:
        steps: list[Step] = []
        turn = self.model.send_user(question)
        model_turns = 1

        while turn.tool_calls:
            if model_turns > self.max_steps:
                return AgentResult(
                    question,
                    "I could not finish the analysis within the step limit. "
                    "Try asking a narrower question.",
                    steps,
                    model_turns,
                    stopped_by_limit=True,
                )
            results = []
            for call in turn.tool_calls:
                start = time.perf_counter()
                result = self.tools.call(call.name, call.args)
                steps.append(Step(call.name, call.args, result, round(time.perf_counter() - start, 4)))
                results.append((call, result))
            turn = self.model.send_tool_results(results)
            model_turns += 1

        answer = (turn.text or "").strip() or "(the model returned an empty answer)"
        return AgentResult(question, answer, steps, model_turns)


def save_log(result: AgentResult, directory: str | Path = "logs") -> Path:
    """Appends one JSON line per question to logs/agent.jsonl."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "agent.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result.to_dict(), ensure_ascii=False, default=str) + "\n")
    return path
