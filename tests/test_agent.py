"""Agent loop tests with a scripted fake model (no network, no API key)."""

from google.genai import types

from arena_agent.agent import ArenaAgent, ModelTurn, ToolCall, save_log
from arena_agent.gemini import GeminiModel, parse_response, to_function_declarations
from arena_agent.tools import TOOL_SPECS


class ScriptedModel:
    """Returns pre-written turns and records what the loop sent back."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.received = []
        self.resets = 0

    def reset(self, system_prompt, tool_specs):
        self.resets += 1
        self.system_prompt = system_prompt
        self.tool_specs = tool_specs

    def send_user(self, text):
        self.received.append(("user", text))
        return self.turns.pop(0)

    def send_tool_results(self, results):
        self.received.append(("tools", results))
        return self.turns.pop(0)


def test_answer_without_tools(tools):
    model = ScriptedModel([ModelTurn("Hello")])
    result = ArenaAgent(model, tools).ask("hi")
    assert result.answer == "Hello"
    assert result.steps == []
    assert result.model_turns == 1


def test_tool_result_is_sent_back_to_model(tools):
    model = ScriptedModel(
        [
            ModelTurn(None, [ToolCall("champion_stats", {"champion": "jinx"}, id="c1")]),
            ModelTurn("Jinx: 4 games, average placement 4.0."),
        ]
    )
    result = ArenaAgent(model, tools).ask("How is Jinx doing?")

    assert result.answer.startswith("Jinx: 4 games")
    assert [s.tool for s in result.steps] == ["champion_stats"]
    assert result.steps[0].result["games"] == 4
    kind, sent = model.received[1]
    assert kind == "tools"
    call, payload = sent[0]
    assert call.id == "c1" and payload["avg_placement"] == 4.0


def test_parallel_calls_go_back_in_one_turn(tools):
    model = ScriptedModel(
        [
            ModelTurn(None, [ToolCall("champion_stats", {"champion": "Jinx"}), ToolCall("champion_stats", {"champion": "Jhin"})]),
            ModelTurn("done"),
        ]
    )
    result = ArenaAgent(model, tools).ask("Jinx vs Jhin")
    assert len(result.steps) == 2
    assert len(model.received[1][1]) == 2
    assert result.model_turns == 2


def test_error_from_tool_reaches_model_and_loop_continues(tools):
    model = ScriptedModel(
        [
            ModelTurn(None, [ToolCall("champion_stats", {"champion": "Jinxx"})]),
            ModelTurn(None, [ToolCall("find_name", {"text": "Jinxx", "kind": "champion"})]),
            ModelTurn(None, [ToolCall("champion_stats", {"champion": "Jinx"})]),
            ModelTurn("Jinx has 4 games."),
        ]
    )
    result = ArenaAgent(model, tools).ask("Jinxx?")
    assert "error" in result.steps[0].result
    assert result.steps[1].result["matches"][0]["name"] == "Jinx"
    assert result.steps[2].result["games"] == 4
    assert result.answer == "Jinx has 4 games."


def test_unknown_tool_and_bad_args_do_not_crash(tools):
    model = ScriptedModel(
        [
            ModelTurn(None, [ToolCall("drop_tables", {}), ToolCall("champion_stats", {"hero": "Jinx"})]),
            ModelTurn("sorry"),
        ]
    )
    result = ArenaAgent(model, tools).ask("x")
    assert all("error" in s.result for s in result.steps)
    assert result.answer == "sorry"


def test_step_limit_stops_endless_tool_calls(tools):
    loop_turn = ModelTurn(None, [ToolCall("find_name", {"text": "a"})])
    model = ScriptedModel([loop_turn] * 10)
    result = ArenaAgent(model, tools, max_steps=3).ask("x")
    assert result.stopped_by_limit
    assert len(result.steps) == 3


def test_empty_answer_is_marked(tools):
    result = ArenaAgent(ScriptedModel([ModelTurn("  ")]), tools).ask("x")
    assert "empty" in result.answer


def test_reset_passes_prompt_and_all_tools(tools):
    model = ScriptedModel([])
    agent = ArenaAgent(model, tools)
    agent.reset()
    assert model.resets == 2
    assert "ONLY" in model.system_prompt
    assert [t["name"] for t in model.tool_specs] == list(tools.registry())


def test_save_log_writes_jsonl(tools, tmp_path):
    result = ArenaAgent(ScriptedModel([ModelTurn("ok")]), tools).ask("Żółw?")
    path = save_log(result, tmp_path)
    line = path.read_text(encoding="utf-8").strip()
    assert '"question": "Żółw?"' in line


# ------------------------------------------------------------ Gemini adapter
def test_function_declarations_cover_all_tools():
    decls = to_function_declarations(TOOL_SPECS)
    assert [d.name for d in decls] == [s["name"] for s in TOOL_SPECS]
    assert decls[0].parameters_json_schema["required"] == ["champion"]


def _response(*parts):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))]
    )


def test_parse_response_reads_calls_and_skips_thoughts():
    call = types.Part(function_call=types.FunctionCall(name="champion_stats", args={"champion": "Jinx"}, id="x1"))
    thought = types.Part(text="thinking...", thought=True)
    turn = parse_response(_response(thought, call))
    assert turn.text is None
    assert turn.tool_calls == [ToolCall("champion_stats", {"champion": "Jinx"}, "x1")]

    turn = parse_response(_response(types.Part(text="Jinx "), types.Part(text="is fine.")))
    assert turn.text == "Jinx is fine." and turn.tool_calls == []


def test_parse_response_without_candidates():
    assert parse_response(types.GenerateContentResponse(candidates=[])).text is None


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(list(contents))
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.models = FakeModels(responses)


def test_gemini_adapter_end_to_end_with_fake_client(tools):
    call = types.Part(function_call=types.FunctionCall(name="champion_stats", args={"champion": "Jinx"}, id="c1"))
    client = FakeClient([_response(call), _response(types.Part(text="Jinx: 4 games."))])
    model = GeminiModel(model="test-model", client=client)
    result = ArenaAgent(model, tools).ask("Jinx?")

    assert result.answer == "Jinx: 4 games."
    # Second request = user question, model's call, our function response.
    second = client.models.calls[1]
    assert [c.role for c in second] == ["user", "model", "user"]
    fr = second[2].parts[0].function_response
    assert fr.name == "champion_stats" and fr.id == "c1" and fr.response["games"] == 4
    # The config must not let the SDK run functions by itself.
    assert model.config.automatic_function_calling.disable is True
