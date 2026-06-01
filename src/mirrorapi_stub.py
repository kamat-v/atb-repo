"""
mirrorapi_stub.py
=================
Deterministic stand-in for the MirrorAPI tool-response server.

In AmbiguousToolBench, success is measured by whether the agent's *tool call
arguments* match the gold values — not by whether the tool's response is
realistic. So we don't need to actually simulate the API behaviour. We just
need to return *something* parseable so the agent's reasoning loop doesn't
crash on a malformed tool response.

The stub is purely a function of (tool_name, arguments). It does no I/O,
makes no network calls, and never raises (unless the tool name is unknown).
"""

from __future__ import annotations
import hashlib
import json
from typing import Any


# A small library of plausible-shaped responses keyed by tool-name suffix.
# When the agent calls a tool, we look at the tool's name suffix and return
# something shaped like a sensible response.
RESPONSE_TEMPLATES = {
    "weather": {"temperature_c": 18, "condition": "clear", "humidity": 0.65},
    "forecast": {"temperature_c": 18, "condition": "clear", "humidity": 0.65},
    "status": {"status": "active", "last_updated": "2026-05-13T12:00:00Z"},
    "search": {"results": [{"title": "Result 1", "url": "https://example.com/1"}]},
    "get": {"data": {"id": "X", "name": "Sample"}},
    "list": {"items": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]},
    "send": {"sent": True, "message_id": "msg_abc123"},
    "cancel": {"cancelled": True, "refund_amount": 0},
    "change": {"changed": True, "previous_value": None, "new_value": None},
    "fetch": {"data": "stub fetched data"},
    "info": {"info": "stub info string"},
    "lookup": {"found": True, "result": "stub lookup result"},
    "check": {"checked": True, "ok": True},
    "post": {"posted": True, "id": "post_abc123"},
    "save": {"saved": True, "path": "/tmp/saved"},
    "export": {"exported": True, "url": "https://example.com/export"},
    "remove": {"removed": True},
    "delete": {"deleted": True},
    "update": {"updated": True},
}


def _pick_template(tool_name: str) -> dict:
    """Pick a template by matching keywords in the tool name."""
    name_lower = tool_name.lower()
    # Strip Tool__API prefix if present
    api_part = name_lower.split("__", 1)[1] if "__" in name_lower else name_lower
    # Check each template keyword
    for keyword, template in RESPONSE_TEMPLATES.items():
        if keyword in api_part:
            return dict(template)  # shallow copy so we can mutate
    # Default
    return {"result": "ok", "data": {}}


def call_tool(tool_name: str, arguments: dict) -> dict:
    """Return a deterministic stub response for the given (tool, args) pair.

    The response is JSON-serialisable. Same inputs always return the same
    output (no rng, no clock). Includes the called tool_name and arguments
    in the response envelope so the agent can verify what was called.
    """
    template = _pick_template(tool_name)
    # Deterministic 'id' field from the args hash so each call gets a
    # distinguishable response if the agent calls the same tool twice with
    # different args.
    args_blob = json.dumps(arguments, sort_keys=True, default=str)
    args_hash = hashlib.sha256(args_blob.encode()).hexdigest()[:8]

    return {
        "ok": True,
        "tool_name": tool_name,
        "echo_arguments": arguments,
        "response_id": args_hash,
        "data": template,
    }


def format_response(resp: dict) -> str:
    """Format a stub response as a JSON string for the LLM to consume."""
    return json.dumps(resp, indent=2)
