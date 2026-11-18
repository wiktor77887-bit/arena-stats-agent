"""Gemini adapter for the agent loop (google-genai SDK).

Automatic function calling is turned off: the SDK must not run Python functions on its own.
The loop in agent.py decides what runs, which keeps every step visible and testable.
"""

from __future__ import annotations

import os
import time
from typing import Any

from google import genai
from google.genai import errors, types

from .agent import ModelTurn, ToolCall

DEFAULT_MODEL = "gemini-3.8-flash"
RETRY_CODES = {429, 500, 503}


def to_function_declarations(tool_specs: list[dict[str, Any]]) -> list[types.FunctionDeclaration]:
    return [
        types.FunctionDeclaration(
            name=spec["name"],
            description=spec["description"],
            parameters_json_schema=spec["parameters"],
        )
        for spec in tool_specs
    ]


def parse_response(response: types.GenerateContentResponse) -> ModelTurn:
    """Pulls text and function calls out of a Gemini response."""
    if not response.candidates or response.candidates[0].content is None:
        return ModelTurn(text=None)
    texts, calls = [], []
    for part in response.candidates[0].content.parts or []:
        if part.function_call:
            fc = part.function_call
            calls.append(ToolCall(name=fc.name, args=dict(fc.args or {}), id=fc.id))
        elif part.text and not part.thought:
            texts.append(part.text)
    return ModelTurn(text="".join(texts) or None, tool_calls=calls)


class GeminiModel:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        temperature: float = 0.0,
        retries: int = 2,
    ):
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
        if client is None:
            key = api_key or os.getenv("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not set. Put it in .env (see .env.example).")
            client = genai.Client(api_key=key)
        self.client = client
        self.temperature = temperature
        self.retries = retries
        self.history: list[types.Content] = []
        self.config: types.GenerateContentConfig | None = None

    # ------------------------------------------------------------ ChatModel
    def reset(self, system_prompt: str, tool_specs: list[dict[str, Any]]) -> None:
        self.history = []
        self.config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[types.Tool(function_declarations=to_function_declarations(tool_specs))],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=self.temperature,
        )

    def send_user(self, text: str) -> ModelTurn:
        self.history.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))
        return self._generate()

    def send_tool_results(self, results: list[tuple[ToolCall, dict[str, Any]]]) -> ModelTurn:
        parts = []
        for call, result in results:
            part = types.Part.from_function_response(name=call.name, response=result)
            part.function_response.id = call.id
            parts.append(part)
        self.history.append(types.Content(role="user", parts=parts))
        return self._generate()

    # ------------------------------------------------------------ internals
    def _generate(self) -> ModelTurn:
        response = self._call_with_retry()
        if response.candidates and response.candidates[0].content is not None:
            # Keep the model's own turn as-is (including thought signatures),
            # Gemini needs it to continue the function-calling exchange.
            self.history.append(response.candidates[0].content)
        return parse_response(response)

    def _call_with_retry(self) -> types.GenerateContentResponse:
        for attempt in range(self.retries + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model, contents=self.history, config=self.config
                )
            except errors.APIError as exc:
                if exc.code not in RETRY_CODES or attempt == self.retries:
                    raise
                time.sleep(2 ** attempt * 2)
        raise AssertionError("unreachable")
