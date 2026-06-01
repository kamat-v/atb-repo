# Architecture

This document specifies the code layout and the contract between the runner, agents, and evaluator. The paper covers what the benchmark *does*; this covers how the code is wired.

## Repository layout

```
atb/
├── data/
│   ├── atb_v1.jsonl              # 87 noisy single-turn instances
│   └── atb_clean_v1.jsonl        # 100 clean instances
├── src/
│   ├── construct_dataset.py      # build dataset from BFCL
│   ├── noise_transforms.py       # MISSING_REQUIRED & AMBIGUOUS_REFERENCE
│   ├── mirrorapi_stub.py         # deterministic oracle simulator
│   ├── runner.py                 # runs (agent, instance) pairs, writes JSON
│   ├── evaluate.py               # aggregates PMR, FSR, TCA, CP, RRAC, OverAsk
│   └── agents/
│       ├── base.py               # _BaseAgent, Action, RunResult, UserSimulator
│       ├── plain.py              # PlainToolCalling
│       ├── react.py              # ReAct
│       ├── heuristic.py          # HeuristicClarify
│       ├── prompted.py           # PromptedClarify
│       └── structural_fep.py     # StructuralFEP
├── notebooks/
│   ├── run_atb.ipynb             # one-click Colab reproduction
│   └── inspect_results.ipynb     # exploratory analysis
├── results/
│   └── {model-slug}/             # per-agent JSONs + summary.csv
├── tests/                        # smoke tests
├── paper/                        # workshop paper
└── docs/                         # this directory
```

## Instance schema

Each line of `data/atb_v1.jsonl` is a JSON record:

```python
{
  "instance_id":                   "simple_69__MISSING_REQUIRED__location",
  "source_query_id":               "simple_69",
  "noise_category":                "MISSING_REQUIRED",
  "noisy_query":                   "Find me some...",
  "original_query":                "Find me a coffee shop in Tokyo...",
  "tool_registry":                 [ ... BFCL function defs ... ],
  "gold_tool_name":                "find_restaurant",
  "gold_arguments":                {"location": "Tokyo", ...},
  "parameter_schema":              { ... OpenAI-format ... },
  "noisy_parameter":               "location",
  "noisy_parameter_gold_value":    "Tokyo",
  "expected_clarification_target": "location"
}
```

Clean instances (`atb_clean_v1.jsonl`) omit the noise-specific fields and use `query` instead of `noisy_query`.

## Agent contract

All agents subclass `_BaseAgent` in `src/agents/base.py`. The contract is one method:

```python
class _BaseAgent:
    name: str  # registered in agents/__init__.py
    
    def _decide_next_action(self, messages, state) -> Action:
        """
        messages: chat history so far (OpenAI format)
        state:    a mutable dict the agent can use to track state across turns
        
        Returns: Action(kind="tool_call", tool_calls=[...]) | 
                 Action(kind="respond", content="...") | 
                 Action(kind="terminate")
        """
```

The runner loops the agent for up to `max_turns` (default 8), dispatching tool calls to a deterministic stub and clarification responses to the oracle user simulator. The full transcript and the final tool call are persisted to a JSON file.

## Runner

`src/runner.py` exposes `run_all(instances_path, results_dir, model, provider, max_turns)`.

For each (instance, agent) pair it:
1. Constructs the agent with `model` and `tool_registry` from the instance.
2. Seeds `messages` with `[{role: "user", content: noisy_query}]`.
3. Loops: call `agent._decide_next_action()`, dispatch the action (tool calls go to `mirrorapi_stub`, responses go to the oracle simulator), append outcome to messages, repeat until terminate or `max_turns`.
4. Writes `results_dir/{agent.name}/{instance_id}.json` with the transcript, final tool call, and metrics.

Resume-safe: skips any (agent, instance) pair whose JSON already exists. Useful when interrupted runs need to be resumed.

## Oracle user simulator

`src/agents/base.py` defines `UserSimulator`, which the runner instantiates per-instance. When an agent emits `Action(kind="respond", content="...?")`, the simulator returns the gold value of `noisy_parameter` regardless of how the question was phrased. This isolates "asking decision" from "question quality" — see the paper §4.3 for the tradeoff discussion.

## Evaluator

`src/evaluate.py` reads all `results_dir/{agent}/*.json` and computes seven metrics per agent (PMR, FSR, TCA, CP, RRAC, OverAsk, UTC), plus per-category breakdowns and pairwise McNemar tests for the headline agent against each baseline.

Output is a CSV with one row per agent. See `results/README.md` for the schema.
