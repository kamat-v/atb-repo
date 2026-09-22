# AmbiguousToolBench (ATB)

A small, controlled benchmark for measuring whether tool-using LLM agents ask for clarification when the user query under-specifies a required argument. Five reference agents, two splits (noisy + clean), six metrics. Designed for cross-model comparison.

See `paper/main.pdf` for the methodology and headline results on Qwen-2.5-14B and Llama-3.1-8B.

---

## Quick start (local GPU machine)

This repo runs locally on any machine with a CUDA GPU. No Colab, no Google Drive, no notebook required.

### 1. Install

```bash
git clone https://github.com/YOUR-ORG/atb.git
cd atb
pip install -r requirements.txt
```

`requirements.txt` pins vLLM `>=0.6.3` and all evaluator/agent dependencies. For AWQ-quantized models, also install `autoawq`:

```bash
pip install autoawq
```

### 2. Launch vLLM

In one terminal:

```bash
./start_vllm.sh Qwen/Qwen2.5-14B-Instruct
```

The script picks the right `--tool-call-parser` automatically (`hermes` for Qwen, `llama3_json` for Llama, `mistral` for Mistral), detects GPU count for tensor parallelism, and detects quantization from the model name (`AWQ`/`GPTQ` suffixes). Wait for `Uvicorn running on http://0.0.0.0:8000` before continuing.

If the model is large for your VRAM, pass through extra vLLM args:

```bash
./start_vllm.sh hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.95
```

### 3. Run the benchmark

In another terminal:

```bash
python run.py --model Qwen/Qwen2.5-14B-Instruct
```

This runs all five agents on the 87 noisy + 100 clean instances, writes per-instance JSON trajectories to `results/qwen2.5-14b-instruct/<agent>/`, and produces `results/qwen2.5-14b-instruct/summary.csv` with the six-metric summary. Resume-safe: re-running is a no-op for completed (agent, instance) pairs.

Wall time is roughly:
- Qwen-2.5-14B on A100-80GB: ~45 minutes.
- Llama-3.1-8B on A100-40GB: ~30 minutes.
- Llama-3.1-70B-AWQ on A100-80GB: ~2-3 hours.
---

## Common run.py recipes

```bash
# All five agents, noisy + clean + evaluate (default)
python run.py --model Qwen/Qwen2.5-14B-Instruct

# Just one agent (useful for iterating on a single design)
python run.py --model Qwen/Qwen2.5-14B-Instruct --agents StructuralFEP

# Skip the clean-instances phase (faster, but no OverAsk metric)
python run.py --model Qwen/Qwen2.5-14B-Instruct --no-clean-split

# Re-evaluate without re-running (after fixing the evaluator, say)
python run.py --model Qwen/Qwen2.5-14B-Instruct --no-run

# Custom output directory
python run.py --model Qwen/Qwen2.5-14B-Instruct --results-dir /data/atb-runs

# Point at any other OpenAI-compatible endpoint
python run.py --model gpt-4o-mini --base-url https://api.openai.com/v1 \
    --api-key $OPENAI_API_KEY
```

Full options: `python run.py --help`.

---

## Repo layout

```
atb/
├── README.md
├── LICENSE                       (MIT)
├── requirements.txt
├── run.py                        Single CLI entry point — start here
├── start_vllm.sh                 Launches vLLM with the right tool parser
│
├── data/
│   ├── atb_v1.jsonl              87 noisy instances
│   └── atb_clean_v1.jsonl        100 clean instances
│
├── src/
│   ├── agents/
│   │   ├── plain.py              PlainToolCalling (no clarification)
│   │   ├── react.py              ReAct (LLM reasoning)
│   │   ├── heuristic.py          HeuristicClarify (rule-based)
│   │   ├── prompted.py           PromptedClarify (LLM prompted to ask)
│   │   ├── structural_fep.py     StructuralFEP (gating rule + LLM uncertainty)
│   │   └── base.py               UserSimulator, RunResult
│   ├── runner.py                 (agent, instance) loop with resume support
│   ├── evaluate.py               Computes seven metrics + writes summary.csv
│   ├── construct_dataset.py      Reproduces atb_v1.jsonl from BFCL
│   └── noise_transforms.py       MISSING_REQUIRED / AMBIGUOUS_REFERENCE / WRONG_PREMISE
│
├── results/
│   ├── qwen2.5-14b-instruct/summary.csv     Frozen results from paper
│   └── llama-3.1-8b-instruct/summary.csv    Frozen results from paper
│
├── tests/                        pytest-discoverable smoke tests
├── docs/
│   ├── architecture.md           How the agents work
│   └── benchmark_spec.md         Dataset construction details
└── paper/
    └── main.{tex,pdf}            The paper (v1.1.1)
```

---

## What gets measured

Six metrics per agent (defined formally in the paper, §B):

| Metric | What it measures |
|---|---|
| PMR | Parameter Match Rate — gold tool called and noisy parameter exactly matches gold value (**headline metric**) |
| FSR | Final Success Rate — stricter PMR requiring all required parameters to match |
| TCA | Tool Choice Accuracy — gold tool, ignoring arguments (the metric BFCL reports) |
| CP | Clarification Precision — fraction of ask-instances where the agent asked about the right parameter |
| RRAC | Recovery Rate Among Clarified — fraction of asks that recovered the gold value |
| OverAsk | Fraction of *clean* instances on which the agent asked unnecessarily |

The evaluator also computes Wilson 95% confidence intervals on PMR and runs McNemar pairwise tests against StructuralFEP for statistical significance.

---

## Adding a new model

If your model has a vLLM tool-call parser, just pass its HuggingFace name:

```bash
./start_vllm.sh mistralai/Mistral-Small-Instruct-2409
python run.py --model mistralai/Mistral-Small-Instruct-2409
```

If `start_vllm.sh` doesn't recognize the family, pass the parser manually:

```bash
./start_vllm.sh your/model --tool-call-parser llama3_json
```

Currently supported vLLM tool parsers: `hermes`, `llama3_json`, `mistral`, `internlm`, `jamba`, `granite`. Check `vllm/entrypoints/openai/tool_parsers/` in your vLLM install for the full list.

---

## Reproducing the paper's results

The two committed `summary.csv` files in `results/` are the exact numbers in the paper's Table 2. To reproduce:

```bash
# Qwen-2.5-14B
./start_vllm.sh Qwen/Qwen2.5-14B-Instruct
python run.py --model Qwen/Qwen2.5-14B-Instruct

# Llama-3.1-8B
./start_vllm.sh meta-llama/Llama-3.1-8B-Instruct
python run.py --model meta-llama/Llama-3.1-8B-Instruct
```

Settings (already encoded in the agents and `start_vllm.sh`): greedy decoding (`temperature=0`), bf16, max 8 turns per (agent, instance), BFCL v3 at commit `70b6a4a`.

---

## Troubleshooting

**`Cannot connect to http://localhost:8000`** — vLLM isn't running. Start it with `./start_vllm.sh <model>` in another terminal.

**`No available memory for the cache blocks`** during vLLM startup — your GPU's KV cache doesn't fit. Reduce `--max-model-len`:
```bash
./start_vllm.sh <model> --max-model-len 2048
```

**`tool_call_parser not found`** — your model family isn't auto-detected. Pass the parser manually:
```bash
./start_vllm.sh <model> --tool-call-parser <hermes|llama3_json|mistral>
```

**Slow runs** — make sure vLLM is using the GPU. Check `nvidia-smi` while running. If the GPU is idle, vLLM probably crashed; check its terminal output.

**`ModuleNotFoundError: No module named 'runner'`** — you're invoking the evaluator or runner directly without `src/` on the path. Use `python run.py` instead; it sets up paths correctly.

---

## Citation

To be added after de-anonymisation.

---

## License

MIT. See `LICENSE`.
