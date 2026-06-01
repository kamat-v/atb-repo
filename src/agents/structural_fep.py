"""
agents/structural_fep.py
========================
StructuralFEPAgent: the FEP-aligned tool-calling agent. Maintains an explicit
joint BeliefState over (tool, parameters); uses the structural rule to inject
a clarifying question when belief entropy exceeds τ.

Loop on each turn:
  1. Get the LLM's proposed tool call (standard tool-calling LLM call).
  2. Build a BeliefState by asking the LLM to enumerate alternative plausible
     values for each parameter of the proposed tool (one extra LLM call,
     returning a small JSON list per parameter).
  3. Compute joint entropy H(tool) + Σ_p H(p|tool).
  4. If H > τ AND the highest-uncertainty parameter has not been clarified
     yet, ask the user a targeted question about that parameter. Skip the
     tool call this turn. Append user reply to messages.
  5. Else: execute the LLM's proposed tool call.

After the user replies to a clarification, the relevant parameter's
distribution collapses (handled implicitly: on the next turn, the LLM sees
the clarification and the corresponding entropy drops).

Per spec §6: τ = 0.5 nats, fixed. No post-hoc tuning.
"""

from __future__ import annotations
import json
import math
import re
from .base import _BaseAgent, Action, llm_complete, message_to_action


# Fixed hyperparameter per spec §6
TAU = 0.5


ENUMERATION_PROMPT = (
    "Given the user's request and the tool call I am about to make, "
    "enumerate the top alternative plausible values for each parameter. "
    "For each parameter, list up to 3 alternative values that a reasonable "
    "person might have meant. If the user's request fully specifies the "
    "parameter, return only the single value with no alternatives.\n\n"
    "Respond ONLY with JSON of the form:\n"
    "{{\"<param_name>\": [\"value1\", \"value2\", ...], ...}}\n\n"
    "User request: {query}\n"
    "Proposed tool: {tool_name}\n"
    "Proposed arguments: {args}\n\n"
    "JSON response:"
)


def _enumerate_alternatives(
    model: str, provider: str, query: str, tool_name: str, arguments: dict,
) -> dict[str, list]:
    """Ask the LLM to enumerate alternative parameter values. Returns
    {param_name: [candidate_values]} dict, or {} on parse failure."""
    prompt = ENUMERATION_PROMPT.format(
        query=query, tool_name=tool_name,
        args=json.dumps(arguments),
    )
    msg = llm_complete(
        model, provider,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=300,
    )
    text = (msg.get("content") or "").strip()
    # Extract JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    # Normalise: each value should be a list
    out = {}
    for k, v in data.items():
        if isinstance(v, list):
            out[k] = v
        else:
            out[k] = [v]
    return out


def _belief_entropy(alternatives: dict[str, list]) -> float:
    """Compute belief entropy: sum over parameters of log(n_alternatives).

    A parameter with one alternative (the LLM's proposed value, no
    alternatives) contributes 0 nats. A parameter with k uniform
    alternatives contributes log(k) nats."""
    H = 0.0
    for _, alts in alternatives.items():
        k = max(1, len(alts))
        if k > 1:
            H += math.log(k)
    return H


def _argmax_uncertainty_param(alternatives: dict[str, list]) -> str | None:
    """Return the parameter with the most alternatives."""
    if not alternatives:
        return None
    best = max(alternatives.items(), key=lambda kv: len(kv[1]))
    if len(best[1]) <= 1:
        return None
    return best[0]


def _clarifying_question(param: str, alternatives: list) -> str:
    """Build a clarifying question targeting the given parameter."""
    # Render param name as natural language: order_id -> "order ID"
    nice = param.replace("_", " ")
    nice = re.sub(r"\bid\b", "ID", nice, flags=re.IGNORECASE)
    opts_str = ", ".join(f"'{a}'" for a in alternatives[:3])
    return (
        f"Which {nice} did you mean? For example, did you mean "
        f"{opts_str}? Please let me know."
    )


class StructuralFEPAgent(_BaseAgent):
    name = "StructuralFEP"

    SYSTEM_PROMPT = (
        "You are a tool-calling assistant. Given a user's request, call the "
        "appropriate tool with the correct arguments. Make your best inference "
        "about parameter values from the user's request."
    )

    def __init__(self, *args, tau: float = TAU, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau

    def _decide_next_action(self, messages, state):
        # First, ask the LLM for its proposed tool call
        msg = llm_complete(
            self.model, self.provider, messages,
            tools=self.tool_registry, temperature=self.temperature,
        )
        action = message_to_action(msg)

        # If the LLM didn't propose a tool call, no structural rule fires
        if action.kind != "tool_call":
            return action

        # If we've already asked enough clarifications, just proceed
        n_clar = len(state.get("asked_clarifications", []))
        if n_clar >= 2:
            return action

        # Build belief over alternatives for each parameter of the proposed call
        # Get the user's original query (first user message)
        user_query = ""
        for m in messages:
            if m.get("role") == "user":
                user_query = m.get("content", "")
                break

        alternatives = _enumerate_alternatives(
            self.model, self.provider,
            query=user_query, tool_name=action.tool_name,
            arguments=action.arguments,
        )

        # Filter out alternatives that have already been resolved by past
        # clarifications. If a previous clarification asked about parameter X,
        # we don't ask about X again.
        already_asked_params = state.setdefault("clarified_params", set())

        # Pick the highest-uncertainty parameter that we haven't asked about yet
        candidates = {
            k: v for k, v in alternatives.items()
            if k not in already_asked_params and len(v) > 1
        }
        H = _belief_entropy(candidates)
        state["last_belief_entropy"] = H

        if H <= self.tau:
            # Confident enough — proceed with the tool call
            return action

        # Above threshold — find the parameter to ask about
        target_param = _argmax_uncertainty_param(candidates)
        if target_param is None:
            return action  # nothing to ask about
        question = _clarifying_question(target_param, candidates[target_param])
        already_asked_params.add(target_param)

        return Action(kind="respond", content=question)
