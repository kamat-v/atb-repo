"""
Smoke-test the noise transforms by running each on a hand-crafted source
and printing the noisy output. The goal is to eyeball that:
  - Each transform produces a grammatical-ish output
  - The expected parameter is the one targeted
  - The gold value is correctly preserved for evaluator use

Run as a script (manual review of output):
    python tests/smoke_noise.py

Or as pytest (auto-discovery, conftest handles path setup):
    pytest tests/
"""

# Path setup is handled by tests/conftest.py when run via pytest.
# When run as a script, fall back to adding src/ relative to this file.
import sys
from pathlib import Path
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from noise_transforms import (
    CleanSource, apply_noise, ELIGIBILITY,
)


# ---------------------------------------------------------------------------
# Fixtures: one for each category
# ---------------------------------------------------------------------------
def fixture_for_missing():
    return CleanSource(
        source_query_id="t1",
        query="Get the weather in Paris tomorrow",
        tool_registry=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "day": {"type": "string"},
                    },
                    "required": ["location"],
                },
            },
        }],
        gold_tool_name="get_weather",
        gold_arguments={"location": "Paris", "day": "tomorrow"},
        parameter_schema={
            "location": {"type": "string"},
            "day": {"type": "string"},
        },
    )


def fixture_for_ambiguous():
    return CleanSource(
        source_query_id="t2",
        query="Cancel order W12345 for me",
        tool_registry=[{
            "type": "function",
            "function": {
                "name": "cancel_order",
                "description": "Cancel order",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        }],
        gold_tool_name="cancel_order",
        gold_arguments={"order_id": "W12345"},
        parameter_schema={"order_id": {"type": "string"}},
    )


def fixture_for_wrong_premise():
    return CleanSource(
        source_query_id="t3",
        query="Change my Wednesday flight to economy class",
        tool_registry=[
            {
                "type": "function",
                "function": {
                    "name": "change_flight_class",
                    "description": "Change class of a flight",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "flight_day": {"type": "string",
                                           "enum": ["Monday", "Tuesday", "Wednesday",
                                                    "Thursday", "Friday"]},
                            "new_class": {"type": "string"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_flight_status",
                    "description": "Look up a flight",
                    "parameters": {
                        "type": "object",
                        "properties": {"flight_day": {"type": "string"}},
                    },
                },
            },
        ],
        gold_tool_name="change_flight_class",
        gold_arguments={"flight_day": "Wednesday", "new_class": "economy"},
        parameter_schema={
            "flight_day": {"type": "string",
                           "enum": ["Monday", "Tuesday", "Wednesday",
                                    "Thursday", "Friday"]},
            "new_class": {"type": "string"},
        },
    )


def fixture_for_underspecified_type():
    return CleanSource(
        source_query_id="t4",
        query="Email the quarterly report to Alex",
        tool_registry=[
            {"type": "function", "function": {"name": "send_email", "description": "..."}},
            {"type": "function", "function": {"name": "send_sms", "description": "..."}},
            {"type": "function", "function": {"name": "send_slack", "description": "..."}},
        ],
        gold_tool_name="send_email",
        gold_arguments={"recipient": "Alex", "subject": "quarterly report"},
        parameter_schema={"recipient": {"type": "string"}, "subject": {"type": "string"}},
    )


def fixture_for_volume_scope():
    return CleanSource(
        source_query_id="t5",
        query="Get the status of order W12345",
        tool_registry=[
            {"type": "function", "function": {
                "name": "get_order_status",
                "description": "...",
                "parameters": {"type": "object",
                               "properties": {"order_id": {"type": "string"}}},
            }},
        ],
        gold_tool_name="get_order_status",
        gold_arguments={"order_id": "W12345"},
        parameter_schema={"order_id": {"type": "string"}},
    )


# ---------------------------------------------------------------------------
# Run each
# ---------------------------------------------------------------------------
def run():
    cases = [
        ("MISSING_REQUIRED", fixture_for_missing()),
        ("AMBIGUOUS_REFERENCE", fixture_for_ambiguous()),
        ("WRONG_PREMISE", fixture_for_wrong_premise()),
        # UNDERSPECIFIED_TYPE and VOLUME_SCOPE dropped in v1.1;
        # fixtures preserved for v1.2 re-enablement.
        # ("UNDERSPECIFIED_TYPE", fixture_for_underspecified_type()),
        # ("VOLUME_SCOPE", fixture_for_volume_scope()),
    ]
    for category, source in cases:
        print(f"\n=== {category} ===")
        print(f"Original:  {source.query!r}")
        eligible = ELIGIBILITY[category](source)
        print(f"Eligible params: {eligible}")
        if not eligible:
            print(f"  SKIP (no eligible parameters)")
            continue
        noisy = apply_noise(source, category, seed=42)
        print(f"Noisy:     {noisy.noisy_query!r}")
        print(f"Targeted:  {noisy.noisy_parameter}")
        print(f"Gold val:  {noisy.noisy_parameter_gold_value!r}")
        # Determinism check: same seed should give same output
        noisy2 = apply_noise(source, category, seed=42)
        assert noisy.noisy_query == noisy2.noisy_query, "non-deterministic!"
        print(f"Determinism: OK")


def test_smoke_noise_categories():
    """Pytest entrypoint — wraps run() for auto-discovery. Asserts the
    categories that ship in v1.1 still produce noisy output."""
    run()


if __name__ == "__main__":
    run()
