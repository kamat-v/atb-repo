"""
agents/heuristic.py
===================
HeuristicClarifyAgent: a trivial rule-based baseline that doesn't trust the LLM
to decide when to clarify. Instead, before issuing any tool call, it inspects
the user query and the tool's required parameters; if any required parameter
appears unmentioned (by simple keyword presence), it asks the user about that
parameter.

This is the "obvious thing to try" baseline that any reviewer will ask about.
It tests whether the more complex agents earn their complexity vs. a simple
keyword-presence check.

Rules:
  1. On the first turn, for each required parameter in the gold tool's schema,
     check whether the parameter name (or any of a small set of synonyms
     derived from the name and description) appears as a substring of the
     user query. If any required parameter has no match, ask about it.
  2. Once a clarification answer is received, treat the parameter as resolved
     and proceed to issue the tool call.
  3. If all required parameters appear mentioned, issue the tool call directly.

This is intentionally simple. It does not use the LLM to decide whether to
ask; it uses pure rule-based keyword presence. The LLM is only used to fill
arguments once the agent has decided to call the tool.
"""

from __future__ import annotations
import re
from .base import _BaseAgent, Action, llm_complete, message_to_action


def _keywords_for_param(param_name: str, param_schema: dict) -> list[str]:
    """Derive lightweight keywords for a parameter from its name + description."""
    kws = set()
    # parameter name itself, including subword splits
    kws.add(param_name.lower())
    for part in re.split(r"[_\-\s]+", param_name.lower()):
        if len(part) >= 3:
            kws.add(part)
    # pull a few content words from the description
    desc = param_schema.get("description", "").lower()
    for word in re.findall(r"[a-z]+", desc):
        if len(word) >= 4 and word not in {"with", "this", "that", "from",
                                             "default", "optional", "value",
                                             "specified", "specific", "given"}:
            kws.add(word)
    # cap to avoid spurious matches via overly generic words
    return sorted(kws)[:8]


def _query_mentions_param(query: str, param_name: str, param_schema: dict) -> bool:
    q = query.lower()
    kws = _keywords_for_param(param_name, param_schema)
    return any(kw in q for kw in kws)


class HeuristicClarifyAgent(_BaseAgent):
    name = "HeuristicClarify"

    SYSTEM_PROMPT = (
        "You are a tool-calling assistant. The system will tell you what to do "
        "at each turn. Follow its instructions exactly."
    )

    def _decide_next_action(self, messages, state):
        # State holds: pending_clarifications (list of param names not yet resolved)
        # and original_query (cached on first call).
        if "original_query" not in state:
            state["original_query"] = self._extract_user_query(messages)
            # Identify required params for the (assumed single) tool
            state["pending_clarifications"] = []
            for tool in self.tool_registry:
                fn = tool.get("function", tool)
                params = fn.get("parameters", {})
                required = params.get("required", [])
                properties = params.get("properties", {})
                for p in required:
                    if not _query_mentions_param(
                        state["original_query"], p, properties.get(p, {})
                    ):
                        state["pending_clarifications"].append(p)

        # If we have a pending clarification, ask about the next one
        if state.get("pending_clarifications"):
            next_param = state["pending_clarifications"][0]
            # check if the most recent message is a user reply we should consume
            if messages and messages[-1].get("role") == "user" and \
                    state.get("awaiting_answer"):
                # consume this answer for the param we asked about
                state["pending_clarifications"].pop(0)
                state["awaiting_answer"] = False
                # fall through to re-check whether more params are pending
                if state["pending_clarifications"]:
                    next_param = state["pending_clarifications"][0]
                else:
                    next_param = None

            if next_param is not None:
                state["awaiting_answer"] = True
                return Action(
                    kind="respond",
                    content=f"What value should I use for the {next_param}?"
                )

        # All required params present (or answered): issue the tool call.
        msg = llm_complete(
            self.model, self.provider, messages,
            tools=self.tool_registry, temperature=self.temperature,
        )
        return message_to_action(msg)

    @staticmethod
    def _extract_user_query(messages):
        for m in messages:
            if m.get("role") == "user":
                return m.get("content", "")
        return ""
