"""
agents/prompted.py
==================
PromptedClarifyAgent: vanilla tool-calling + a system-prompt instruction to
ask if unsure. The control that tests whether *prompting alone* captures the
gain of clarification-seeking, or whether structural intervention is needed.
"""

from .base import _BaseAgent, llm_complete, message_to_action


class PromptedClarifyAgent(_BaseAgent):
    name = "PromptedClarify"

    SYSTEM_PROMPT = (
        "You are a tool-calling assistant. Given a user's request, call the "
        "appropriate tool with the correct arguments.\n\n"
        "IMPORTANT: If you are unsure about any parameter value, or about "
        "which tool the user wants you to use, ask the user a clarifying "
        "question BEFORE calling the tool. Phrase your question to identify "
        "exactly which parameter you need. After receiving the user's reply, "
        "proceed with the correct tool call."
    )

    def _decide_next_action(self, messages, state):
        msg = llm_complete(
            self.model, self.provider, messages,
            tools=self.tool_registry, temperature=self.temperature,
        )
        return message_to_action(msg)
