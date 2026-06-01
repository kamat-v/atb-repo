"""
Smoke-test the four agents with a deterministic mock LLM.

The mock simulates two scenarios per agent:
  Scenario A (clear query): the LLM produces a tool call directly
  Scenario B (ambiguous query, FEP enumeration): LLM enumerates alternatives,
    structural rule triggers clarification, then user reply collapses belief
"""

import json
import sys
import types
from pathlib import Path

# Insert src on path
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Mock litellm.completion
# ---------------------------------------------------------------------------
class MockMessage:
    def __init__(self, content="", tool_calls=None):
        self._content = content
        self._tool_calls = tool_calls

    def model_dump(self):
        return {"content": self._content, "tool_calls": self._tool_calls}


class MockChoice:
    def __init__(self, message):
        self.message = message


class MockResp:
    def __init__(self, message):
        self.choices = [MockChoice(message)]


def make_mock_completion(scripted_responses):
    """Build a completion() function that returns the next scripted response
    on each call, looking up by which kind of request we're handling."""
    call_idx = [0]

    def completion(**kwargs):
        idx = call_idx[0]
        call_idx[0] += 1
        # Decide what to return based on whether tools= is in kwargs
        # (tool-calling prompt) vs not (enumeration prompt)
        if "tools" in kwargs and kwargs["tools"]:
            # Tool-calling request — return next scripted tool_call response
            resp = scripted_responses["tool"][idx % len(scripted_responses["tool"])]
            return MockResp(MockMessage(
                content="", tool_calls=[{
                    "id": "tc1", "type": "function",
                    "function": {
                        "name": resp["name"],
                        "arguments": json.dumps(resp["args"]),
                    },
                }],
            ))
        else:
            # Enumeration prompt — return scripted enumeration JSON
            enum = scripted_responses.get("enum", {})
            return MockResp(MockMessage(content=json.dumps(enum)))
    return completion


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
class FakeInstance:
    def __init__(self, instance_id, noisy_query, expected_target, gold_value):
        self.instance_id = instance_id
        self.noisy_query = noisy_query
        self.expected_clarification_target = expected_target
        self.noisy_parameter_gold_value = gold_value


def fixture_clear():
    return FakeInstance(
        instance_id="test_clear",
        noisy_query="Get the weather in Paris tomorrow",
        expected_target="location",
        gold_value="Paris",
    )


def fixture_ambiguous():
    return FakeInstance(
        instance_id="test_ambig",
        noisy_query="Get the weather in there tomorrow",
        expected_target="location",
        gold_value="Paris",
    )


def fake_registry():
    return [{
        "type": "function",
        "function": {
            "name": "Weather__GetForecast",
            "description": "Get weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "day": {"type": "string"},
                },
                "required": ["location", "day"],
            },
        },
    }]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def patch_litellm(mock_completion):
    """Inject our mock into the litellm module BEFORE the agent's import."""
    import litellm  # may not be installed
    litellm.completion = mock_completion


def run():
    # Install a fake litellm module so agents can import
    fake_litellm = types.ModuleType("litellm")
    fake_litellm.completion = None  # placeholder, overridden per test
    sys.modules["litellm"] = fake_litellm

    # Now import the agents
    from agents import (
        PlainToolCallingAgent, ReActAgent,
        PromptedClarifyAgent, StructuralFEPAgent,
    )
    from agents.base import UserSimulator

    # ----- Test 1: Plain agent on clear query -----
    print("=== Test 1: PlainToolCalling, clear query ===")
    fake_litellm.completion = make_mock_completion({
        "tool": [{"name": "Weather__GetForecast",
                  "args": {"location": "Paris", "day": "tomorrow"}}],
    })
    agent = PlainToolCallingAgent(tool_registry=fake_registry(),
                                  model="mock", provider="openai")
    user = UserSimulator("location", "Paris")
    result = agent.solve(fixture_clear(), user, max_turns=4)
    print(f"  final_tool_call: {result.final_tool_call}")
    print(f"  turn_count: {result.turn_count}  error: {result.error}")
    assert result.final_tool_call is not None
    assert result.final_tool_call["arguments"]["location"] == "Paris"

    # ----- Test 2: StructuralFEP on ambiguous query -----
    print("\n=== Test 2: StructuralFEP, ambiguous query ===")
    # LLM proposes a guess; then enumeration returns multiple alternatives
    # for 'location'; FEP asks a clarifying question; user responds "Paris";
    # next turn LLM produces correct tool call.
    call_state = {"step": 0}
    def fep_completion(**kwargs):
        step = call_state["step"]
        call_state["step"] += 1
        if "tools" in kwargs and kwargs["tools"]:
            # Tool call request
            if step == 0:
                # First proposal: guesses location
                return MockResp(MockMessage(content="", tool_calls=[{
                    "id": "tc1", "type": "function",
                    "function": {"name": "Weather__GetForecast",
                                 "arguments": json.dumps({
                                     "location": "London", "day": "tomorrow"})},
                }]))
            else:
                # After clarification: correct call
                return MockResp(MockMessage(content="", tool_calls=[{
                    "id": "tc2", "type": "function",
                    "function": {"name": "Weather__GetForecast",
                                 "arguments": json.dumps({
                                     "location": "Paris", "day": "tomorrow"})},
                }]))
        else:
            # Enumeration request
            return MockResp(MockMessage(content=json.dumps({
                "location": ["London", "Paris", "Berlin"],
                "day": ["tomorrow"],
            })))
    fake_litellm.completion = fep_completion
    agent = StructuralFEPAgent(tool_registry=fake_registry(),
                               model="mock", provider="openai")
    user = UserSimulator("location", "Paris")
    result = agent.solve(fixture_ambiguous(), user, max_turns=4)
    print(f"  asked clarifications: {len(result.transcript) // 2}")
    print(f"  final_tool_call: {result.final_tool_call}")
    print(f"  turn_count: {result.turn_count}")
    # Print transcript
    for m in result.transcript:
        if "tool_call" in m:
            print(f"  TOOL: {m['tool_call']}")
        elif m.get("role") == "assistant":
            print(f"  ASSISTANT: {m.get('content', '')[:100]}")
        elif m.get("role") == "user":
            print(f"  USER: {m.get('content', '')[:100]}")
    assert result.final_tool_call is not None, "FEP must reach a tool call"
    assert result.final_tool_call["arguments"]["location"] == "Paris", \
        f"FEP should recover Paris, got {result.final_tool_call}"

    # ----- Test 3: PromptedClarify, mid behaviour -----
    # The LLM chooses to ask first because the system prompt instructs it to
    print("\n=== Test 3: PromptedClarify, asks then calls ===")
    call_state2 = {"step": 0}
    def prompted_completion(**kwargs):
        step = call_state2["step"]
        call_state2["step"] += 1
        if step == 0:
            # First: text-only clarifying question (no tool call)
            return MockResp(MockMessage(
                content="Which location did you mean?",
                tool_calls=None,
            ))
        else:
            # After clarification: tool call
            return MockResp(MockMessage(content="", tool_calls=[{
                "id": "tc1", "type": "function",
                "function": {"name": "Weather__GetForecast",
                             "arguments": json.dumps({
                                 "location": "Paris", "day": "tomorrow"})},
            }]))
    fake_litellm.completion = prompted_completion
    agent = PromptedClarifyAgent(tool_registry=fake_registry(),
                                 model="mock", provider="openai")
    user = UserSimulator("location", "Paris")
    result = agent.solve(fixture_ambiguous(), user, max_turns=4)
    print(f"  final_tool_call: {result.final_tool_call}")
    print(f"  turn_count: {result.turn_count}")
    assert result.final_tool_call is not None
    assert result.final_tool_call["arguments"]["location"] == "Paris"

    print("\nAll agent smoke tests passed.")


def test_smoke_agents():
    """Pytest entrypoint."""
    run()


if __name__ == "__main__":
    run()
