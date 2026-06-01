#!/usr/bin/env python3
"""
run.py — Single entry point to evaluate AmbiguousToolBench on any model.

Assumes a vLLM OpenAI-compatible server is already running locally. Use the
companion `start_vllm.sh` to launch one, or point at any other OpenAI-compatible
endpoint via --base-url.

Examples
--------

  # 1. Launch vLLM in another terminal (or use start_vllm.sh):
  #    python -m vllm.entrypoints.openai.api_server \
  #      --model Qwen/Qwen2.5-14B-Instruct \
  #      --tool-call-parser hermes --enable-auto-tool-choice

  # 2. Run the full benchmark end-to-end (noisy + clean + evaluate):
  python run.py --model Qwen/Qwen2.5-14B-Instruct

  # 3. Run only one agent (resume-safe — re-running is a no-op):
  python run.py --model Qwen/Qwen2.5-14B-Instruct --agents StructuralFEP

  # 4. Use a custom results directory:
  python run.py --model Qwen/Qwen2.5-14B-Instruct --results-dir /data/atb_runs

  # 5. Skip the evaluation step (run agents only):
  python run.py --model Qwen/Qwen2.5-14B-Instruct --no-eval

  # 6. Skip agent runs and only re-evaluate existing results:
  python run.py --model Qwen/Qwen2.5-14B-Instruct --no-run

  # 7. Point at a remote OpenAI-compatible endpoint:
  python run.py --model gpt-4o-mini --base-url https://api.openai.com/v1 \
      --api-key $OPENAI_API_KEY
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"

DEFAULT_AGENTS = [
    "PlainToolCalling",
    "ReAct",
    "HeuristicClarify",
    "PromptedClarify",
    "StructuralFEP",
]


def model_slug(model: str) -> str:
    """Derive a stable directory name from a model identifier.

    Matches the convention used by previously-released results in
    results/qwen-2.5-14b-instruct/ and results/llama-3.1-8b-instruct/:
    lowercase + slash→hyphen + dots preserved.
    """
    return model.split("/")[-1].lower()


def check_vllm_endpoint(base_url: str) -> None:
    """Verify the OpenAI-compatible endpoint is reachable. Fail fast if not."""
    import requests
    try:
        r = requests.get(f"{base_url.rstrip('/')}/models", timeout=5)
        if r.status_code != 200:
            raise RuntimeError(
                f"Endpoint {base_url}/models returned HTTP {r.status_code}.\n"
                f"Response: {r.text[:300]}"
            )
        served = [m["id"] for m in r.json().get("data", [])]
        print(f"[OK] Endpoint {base_url} responding. Serving: {served}")
    except (requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
            ConnectionError) as e:
        msg = (
            f"Cannot connect to {base_url}.\n"
            f"Underlying error: {type(e).__name__}: {str(e)[:200]}\n\n"
            f"Is vLLM running? Launch it with:\n"
            f"  ./start_vllm.sh <model-name>\n"
            f"or:\n"
            f"  python -m vllm.entrypoints.openai.api_server --model <name> "
            f"--tool-call-parser <hermes|llama3_json> --enable-auto-tool-choice\n\n"
            f"To skip this check (e.g. against a remote endpoint), pass --skip-check."
        )
        # Use SystemExit so we get a clean error message, not a traceback
        print(f"\n[ERROR] {msg}", file=sys.stderr)
        sys.exit(2)


def setup_paths_and_env(base_url: str, api_key: str) -> None:
    """Add src/ to sys.path and set OpenAI env vars."""
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_KEY"] = api_key


def do_run(model: str, results_dir: Path, agents: list[str],
           max_turns: int, instances_path: Path) -> None:
    """Run a set of agents against an instances JSONL file."""
    from runner import run_all  # imported after setup_paths_and_env

    print(f"\n[RUN] model={model} agents={agents}")
    print(f"      instances={instances_path}")
    print(f"      results_dir={results_dir}")

    # The runner expects a litellm-style identifier "openai/<model>" so it
    # routes through the OpenAI provider (which honours OPENAI_BASE_URL).
    routed_model = f"openai/{model}" if not model.startswith("openai/") else model

    run_all(
        instances_path=str(instances_path),
        results_dir=str(results_dir),
        model=routed_model,
        provider="openai",
        max_turns=max_turns,
        agent_names=agents,
    )


def do_eval(model: str, results_dir: Path, agents: list[str]) -> None:
    """Run the evaluator on completed results, write summary.csv."""
    evaluator = SRC_DIR / "evaluate.py"
    summary_csv = results_dir / "summary.csv"

    # Safety: if there are no per-instance JSON results and a summary already
    # exists, refuse to clobber it. This protects users who downloaded a frozen
    # results snapshot and accidentally run --no-run + eval.
    has_per_instance = any(
        (results_dir / agent).exists() and any((results_dir / agent).glob("*.json"))
        for agent in agents
    )
    if not has_per_instance and summary_csv.exists():
        print(f"[SKIP] No per-instance results found in {results_dir}/.")
        print(f"       Existing {summary_csv} preserved (would have been overwritten with zeros).")
        print(f"       To regenerate: run agents first without --no-run, or delete summary.csv.")
        return

    cmd = [
        sys.executable, str(evaluator),
        "--instances", str(DATA_DIR / "atb_v1.jsonl"),
        "--clean-instances", str(DATA_DIR / "atb_clean_v1.jsonl"),
        "--results-dir", str(results_dir),
        "--agents", *agents,
        "--output", str(summary_csv),
    ]

    print(f"\n[EVAL] Running evaluator")
    print(f"       Output: {summary_csv}")
    result = subprocess.run(cmd, cwd=str(SRC_DIR))
    if result.returncode != 0:
        raise RuntimeError(f"Evaluator failed with code {result.returncode}")
    print(f"[OK] Summary written to {summary_csv}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--model", required=True,
        help='Model identifier as served by vLLM (e.g. "Qwen/Qwen2.5-14B-Instruct").',
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000/v1",
        help="OpenAI-compatible endpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--api-key", default="dummy",
        help='API key. Use "dummy" for local vLLM (default).',
    )
    parser.add_argument(
        "--results-dir", default=None,
        help=f"Output directory. Defaults to {DEFAULT_RESULTS_ROOT}/<model-slug>/",
    )
    parser.add_argument(
        "--agents", nargs="+", default=DEFAULT_AGENTS,
        choices=DEFAULT_AGENTS,
        help="Which agents to run (default: all five).",
    )
    parser.add_argument(
        "--max-turns", type=int, default=8,
        help="Maximum turns per (agent, instance) pair (default: %(default)s).",
    )
    parser.add_argument(
        "--no-run", action="store_true",
        help="Skip agent runs; only re-evaluate existing results.",
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Run agents only; skip the evaluator.",
    )
    parser.add_argument(
        "--no-clean-split", action="store_true",
        help="Skip the clean-instances phase (OverAsk metric will be unavailable).",
    )
    parser.add_argument(
        "--skip-check", action="store_true",
        help="Skip the endpoint reachability check.",
    )
    args = parser.parse_args()

    # 1. Resolve paths
    results_dir = (
        Path(args.results_dir) if args.results_dir
        else DEFAULT_RESULTS_ROOT / model_slug(args.model)
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # 2. Check endpoint unless skipped
    if not args.skip_check:
        check_vllm_endpoint(args.base_url)

    # 3. Set up sys.path and env
    setup_paths_and_env(args.base_url, args.api_key)

    # 4. Run agents on noisy and (optionally) clean splits
    if not args.no_run:
        do_run(args.model, results_dir, args.agents,
               args.max_turns, DATA_DIR / "atb_v1.jsonl")
        if not args.no_clean_split:
            do_run(args.model, results_dir, args.agents,
                   args.max_turns, DATA_DIR / "atb_clean_v1.jsonl")

    # 5. Evaluate
    if not args.no_eval:
        do_eval(args.model, results_dir, args.agents)

    print(f"\n[DONE] All results in: {results_dir}")
    if not args.no_eval:
        print(f"       Summary: {results_dir}/summary.csv")


if __name__ == "__main__":
    main()
