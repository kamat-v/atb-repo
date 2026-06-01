"""
agents/react.py
===============
ReActAgent: interleaves natural-language reasoning with tool calls (Yao et al. 2023).

Implementation: we add a small instruction to the system prompt asking the
model to reason briefly before each tool call. The model is still free to
emit a text reasoning step or directly a tool call. This is the de-facto
baseline agent in the community.
"""

from .base import _BaseAgent, llm_complete, message_to_action


class ReActAgent(_BaseAgent):
    name = "ReAct"

    SYSTEM_PROMPT = (
        "You are a tool-calling assistant using the ReAct pattern. Before "
        "each action, briefly reason about what the user wants and which "
        "tool to use. Then call the tool. If your reasoning suggests you "
        "lack a required argument, you may either guess based on the user's "
        "intent or call the most likely tool."
    )

    def _decide_next_action(self, messages, state):
        msg = llm_complete(
            self.model, self.provider, messages,
            tools=self.tool_registry, temperature=self.temperature,
        )
        return message_to_action(msg)
