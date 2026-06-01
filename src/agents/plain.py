"""
agents/plain.py
===============
PlainToolCallingAgent: vanilla tool-calling agent. The floor baseline.

Behaviour: ask the LLM for a tool call, dispatch it, done. No clarification
logic. If the LLM produces a text response instead of a tool call, that
counts as a "finish" with no successful tool call.
"""

from __future__ import annotations
from .base import _BaseAgent, Action, llm_complete, message_to_action


class PlainToolCallingAgent(_BaseAgent):
    name = "PlainToolCalling"

    SYSTEM_PROMPT = (
        "You are a tool-calling assistant. Given a user's request, call the "
        "appropriate tool with the correct arguments. Do not ask the user "
        "for clarification — make your best guess from the request alone."
    )

    def _decide_next_action(self, messages, state):
        msg = llm_complete(
            self.model, self.provider, messages,
            tools=self.tool_registry, temperature=self.temperature,
        )
        return message_to_action(msg)
