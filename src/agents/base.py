"""
agents/base.py
==============
Common agent skeleton. All four agents subclass _BaseAgent and override
_decide_next_action.

The loop:
  1. Initialise messages = [system_prompt, user_query]
  2. While not done and turn < max_turns:
       a. action = self._decide_next_action(messages, state)
       b. if action is RESPOND (a clarifying question to the user):
            user_reply = user.respond(action.content)
            append assistant+user messages
       c. elif action is TOOL_CALL:
            response = mirrorapi_stub.call_tool(action.name, action.args)
            append tool_call + tool_response messages
            mark done if this is a "final" tool call
       d. elif action is FINISH: break
  3. Return RunResult

Each agent overrides:
  - _decide_next_action(messages, state) -> Action
  - _is_final_tool_call(action, state) -> bool (defaults to True)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import json
import time
import traceback


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------
@dataclass
class Action:
    kind: str                       # "tool_call" | "respond" | "finish"
    tool_name: Optional[str] = None
    arguments: dict = field(default_factory=dict)
    content: str = ""               # text for respond, or commentary for tool_call


# ---------------------------------------------------------------------------
# Run result
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    instance_id: str
    agent_name: str
    transcript: list[dict] = field(default_factory=list)
    final_tool_call: Optional[dict] = None  # {'name', 'arguments'}
    turn_count: int = 0
    wall_time_s: float = 0.0
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# User simulator interface (scripted, see benchmark_spec §7)
# ---------------------------------------------------------------------------
class UserSimulator:
    """Scripted user that knows the gold value for the noisy parameter.

    Rules from benchmark_spec §7:
      - If asked a question that references `expected_clarification_target`:
        respond with the gold value in a natural-language template
      - If asked something else: respond "I'm not sure what you mean, please proceed."
      - The user does not interject unless asked
    """

    def __init__(
        self,
        expected_clarification_target: str,
        gold_value: Any,
        synonyms: list[str] = None,
    ):
        self.target = expected_clarification_target
        self.gold = gold_value
        # Build search terms: the target itself + any synonyms
        terms = [expected_clarification_target.lower()]
        # Common variations: order_id -> "order", "id"; user_name -> "user", "name"
        if "_" in expected_clarification_target:
            for part in expected_clarification_target.split("_"):
                if len(part) >= 2:
                    terms.append(part.lower())
        if synonyms:
            terms.extend(s.lower() for s in synonyms)
        # Add common synonyms for the __TOOL__ sentinel
        if expected_clarification_target == "tool_choice":
            terms.extend(["which tool", "what kind", "how would you like",
                          "which medium", "how should i", "what format",
                          "by what means", "delivery method", "channel"])
        self._terms = terms

    def respond(self, agent_message: str) -> str:
        msg = agent_message.lower()
        if any(t in msg for t in self._terms):
            return f"{self.gold}"
        return "I'm not sure what you mean, please proceed."


# ---------------------------------------------------------------------------
# LLM adapter (litellm-based)
# ---------------------------------------------------------------------------
def llm_complete(
    model: str, provider: str, messages: list[dict],
    tools: Optional[list[dict]] = None, temperature: float = 0.0,
    max_tokens: int = 1024,
) -> dict:
    """Call litellm and return a message dict. Caller handles parse + dispatch."""
    from litellm import completion
    kwargs = {
        "model": model,
        "custom_llm_provider": provider,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools is not None:
        kwargs["tools"] = tools
    resp = completion(**kwargs)
    return resp.choices[0].message.model_dump()


def message_to_action(message: dict) -> Action:
    """Convert a litellm response message into our Action enum."""
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        tc = tool_calls[0]
        fn = tc.get("function", {})
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (json.JSONDecodeError, TypeError):
            args = {}
        return Action(kind="tool_call", tool_name=name, arguments=args)
    content = (message.get("content") or "").strip()
    if not content:
        return Action(kind="finish", content="")
    return Action(kind="respond", content=content)


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------
class _BaseAgent:
    name: str = "Base"

    SYSTEM_PROMPT = (
        "You are a tool-calling assistant. Given a user's request, you should "
        "call the appropriate tool with the correct arguments. If you are "
        "unsure about any parameter value, you may ask the user a clarifying "
        "question, but only when truly necessary."
    )

    def __init__(
        self, tool_registry: list[dict], model: str = "openai/Qwen/Qwen2.5-14B-Instruct",
        provider: str = "openai", temperature: float = 0.0,
    ):
        self.tool_registry = tool_registry
        self.model = model
        self.provider = provider
        self.temperature = temperature

    def _decide_next_action(self, messages: list[dict], state: dict) -> Action:
        """Override in subclasses."""
        raise NotImplementedError

    def _system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def solve(self, instance, user: UserSimulator,
              max_turns: int = 8) -> RunResult:
        """Run one episode."""
        from mirrorapi_stub import call_tool, format_response

        t0 = time.time()
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": instance.noisy_query},
        ]
        state = {"turn": 0, "asked_clarifications": []}
        transcript: list[dict] = list(messages)
        final_tool_call = None
        result_extra: dict = {}

        try:
            for turn in range(max_turns):
                state["turn"] = turn
                action = self._decide_next_action(messages, state)

                if action.kind == "finish":
                    break

                if action.kind == "respond":
                    # Agent is asking the user something
                    state["asked_clarifications"].append(action.content)
                    user_reply = user.respond(action.content)
                    messages.append({"role": "assistant",
                                     "content": action.content})
                    messages.append({"role": "user", "content": user_reply})
                    transcript.append({"role": "assistant",
                                       "content": action.content})
                    transcript.append({"role": "user", "content": user_reply})
                    continue

                if action.kind == "tool_call":
                    # Dispatch via stub
                    tool_resp = call_tool(action.tool_name, action.arguments)
                    tool_resp_str = format_response(tool_resp)
                    # Record the call
                    tc_id = f"tc_{turn}"
                    messages.append({
                        "role": "assistant",
                        "tool_calls": [{
                            "id": tc_id, "type": "function",
                            "function": {
                                "name": action.tool_name,
                                "arguments": json.dumps(action.arguments),
                            },
                        }],
                        "content": "",
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": action.tool_name,
                        "content": tool_resp_str,
                    })
                    transcript.append({"role": "assistant",
                                       "tool_call": {"name": action.tool_name,
                                                     "arguments": action.arguments}})
                    transcript.append({"role": "tool",
                                       "name": action.tool_name,
                                       "content": tool_resp_str})
                    # First tool call is the "final" one for evaluation purposes,
                    # *unless* it's a status-lookup tool that the agent might call
                    # for WRONG_PREMISE verification.
                    if final_tool_call is None:
                        if not _is_lookup_tool(action.tool_name):
                            final_tool_call = {
                                "name": action.tool_name,
                                "arguments": action.arguments,
                            }
                            break
                    # If it was a lookup, keep going (agent should now decide)
                    continue
        except Exception as e:
            result_extra["error_trace"] = traceback.format_exc()
            return RunResult(
                instance_id=instance.instance_id, agent_name=self.name,
                transcript=transcript, final_tool_call=final_tool_call,
                turn_count=state["turn"] + 1,
                wall_time_s=time.time() - t0,
                error=f"{type(e).__name__}: {e}",
                extra=result_extra,
            )

        return RunResult(
            instance_id=instance.instance_id, agent_name=self.name,
            transcript=transcript, final_tool_call=final_tool_call,
            turn_count=state["turn"] + 1,
            wall_time_s=time.time() - t0,
            error=None,
            extra=result_extra,
        )


def _is_lookup_tool(name: str) -> bool:
    n = name.lower()
    api_part = n.split("__", 1)[1] if "__" in n else n
    return any(kw in api_part for kw in
               ["check_", "status", "lookup", "info_", "get_status"])
