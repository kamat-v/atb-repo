# Results

This directory holds per-model evaluation outputs:

```
results/
├── qwen-2.5-14b-instruct/
│   ├── summary.csv                # aggregated metrics
│   ├── PlainToolCalling/          # per-instance JSONs
│   ├── ReAct/
│   ├── HeuristicClarify/
│   ├── PromptedClarify/
│   └── StructuralFEP/
└── llama-3.1-8b-instruct/
    └── ... (same layout)
```

## Summary CSV schema

Each row is one agent. Columns:

| Column | Meaning |
|---|---|
| `agent` | Agent name |
| `pmr` | Parameter Match Rate on 80 noisy instances (excludes WRONG_PREMISE pilot) |
| `pmr_ci_low`, `pmr_ci_high` | 95% Wilson confidence interval |
| `fsr` | Final Success Rate (stricter than PMR — all required parameters match) |
| `tca` | Tool Choice Accuracy (gold tool, ignoring arguments) |
| `cp` | Clarification Precision (asked instances only) |
| `rrac` | Recovery Rate Among Clarified |
| `overask` | OverAsk on 100 clean instances |
| `n_asked` | Number of instances on which the agent asked at least once |
| `pmr_ar` | PMR on AMBIGUOUS_REFERENCE subset (n=40) |
| `pmr_mr` | PMR on MISSING_REQUIRED subset (n=40) |

## Per-instance JSON schema

Each file in `{agent}/` is named `{instance_id}.json` and contains the full agent trajectory:

```json
{
  "instance_id": "simple_69__MISSING_REQUIRED__location",
  "agent_name": "StructuralFEP",
  "transcript": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    ...
  ],
  "final_tool_call": {"name": "...", "arguments": {...}},
  "turns": 3,
  "duration_seconds": 2.7,
  "pmr": 1,
  "asked": true,
  "asked_target": "location",
  "extra": {...}
}
```

## Regenerating the summary

If you have the per-instance JSONs but want to recompute `summary.csv`:

```bash
python -m src.evaluate \
    --instances data/atb_v1.jsonl \
    --clean-instances data/atb_clean_v1.jsonl \
    --results-dir results/qwen-2.5-14b-instruct \
    --agents PlainToolCalling ReAct HeuristicClarify PromptedClarify StructuralFEP \
    --output results/qwen-2.5-14b-instruct/summary.csv
```
