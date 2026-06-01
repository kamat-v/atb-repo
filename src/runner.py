"""
runner.py
=========
Drives one (agent, instance) run, writes RunResult to disk. Resume-safe:
on startup scans results/ to skip pairs already completed.

Usage:
    from runner import run_all
    run_all(
        instances_path="data/atb_v1.jsonl",
        agents=[PlainToolCallingAgent, ...],
        results_dir="results",
        model="openai/Qwen/Qwen2.5-14B-Instruct",
        provider="openai",
    )
"""

from __future__ import annotations
import json
import time
from dataclasses import asdict
from pathlib import Path

# Import locally so test runs don't require litellm at import time
def _import_agents():
    from agents import (
        PlainToolCallingAgent, ReActAgent,
        HeuristicClarifyAgent,
        PromptedClarifyAgent, StructuralFEPAgent,
    )
    from agents.base import UserSimulator
    return {
        "PlainToolCalling": PlainToolCallingAgent,
        "ReAct": ReActAgent,
        "HeuristicClarify": HeuristicClarifyAgent,
        "PromptedClarify": PromptedClarifyAgent,
        "StructuralFEP": StructuralFEPAgent,
    }, UserSimulator


def load_instances(path: str) -> list[dict]:
    """Load JSONL of dataset instances."""
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def _normalize_instance(inst: dict) -> dict:
    """Both noisy (NoisyInstance) and clean (CleanSource) records pass through
    here. Returns a dict with these guaranteed fields:
      - instance_id
      - noisy_query                (the user message the agent sees)
      - tool_registry
      - gold_tool_name
      - gold_arguments
      - noisy_parameter            (the parameter the noise targets, or "__CLEAN__")
      - noisy_parameter_gold_value (the gold value, or None for clean)
      - expected_clarification_target (target for clarification, or "__NONE__")
      - noise_category             ("CLEAN" for clean instances)
    """
    out = dict(inst)
    if "instance_id" not in out:
        # Clean instance — synthesise an instance_id from source_query_id
        qid = out.get("source_query_id") or out.get("query_id") or "unknown"
        out["instance_id"] = f"{qid}__CLEAN"
    if "noisy_query" not in out:
        # Clean instance uses 'query' as the user message
        out["noisy_query"] = out.get("query", "")
    if "noise_category" not in out:
        out["noise_category"] = "CLEAN"
    if "noisy_parameter" not in out:
        out["noisy_parameter"] = "__CLEAN__"
    if "noisy_parameter_gold_value" not in out:
        out["noisy_parameter_gold_value"] = None
    if "expected_clarification_target" not in out:
        out["expected_clarification_target"] = "__NONE__"
    return out


def _key(agent_name: str, instance_id: str) -> str:
    safe = instance_id.replace("/", "_").replace(":", "_")
    return f"{agent_name}__{safe}"


def _result_path(results_dir: Path, agent_name: str, instance_id: str) -> Path:
    return results_dir / agent_name / f"{_key(agent_name, instance_id)}.json"


def _completed_keys(results_dir: Path, agent_names: list[str]) -> set[tuple[str, str]]:
    out = set()
    for name in agent_names:
        d = results_dir / name
        if not d.exists():
            continue
        for f in d.glob("*.json"):
            stem = f.stem
            if stem.startswith(name + "__"):
                instance_id = stem[len(name) + 2:]
                out.add((name, instance_id))
    return out


def _instance_to_obj(inst: dict):
    """Build a minimal object with the fields agents need."""
    class _I:
        pass
    i = _I()
    for k, v in inst.items():
        setattr(i, k, v)
    return i


def run_pair(agent_cls, inst_dict: dict, model: str, provider: str,
             max_turns: int = 8):
    """Run one (agent, instance) pair. Returns a RunResult."""
    _, UserSim = _import_agents()
    inst_dict = _normalize_instance(inst_dict)
    inst = _instance_to_obj(inst_dict)
    agent = agent_cls(
        tool_registry=inst_dict["tool_registry"],
        model=model, provider=provider,
    )
    user = UserSim(
        expected_clarification_target=inst_dict["expected_clarification_target"],
        gold_value=inst_dict["noisy_parameter_gold_value"],
    )
    result = agent.solve(inst, user, max_turns=max_turns)
    return result


def run_all(
    instances_path: str, results_dir: str = "results",
    model: str = "openai/Qwen/Qwen2.5-14B-Instruct",
    provider: str = "openai",
    agent_names: list[str] = None,
    max_turns: int = 8,
    verbose: bool = True,
):
    """Run every (agent, instance) pair, with resume support.

    Each completed result is written to {results_dir}/{agent_name}/{key}.json
    immediately, so if the run dies mid-way we can resume.
    """
    agents_map, _ = _import_agents()
    if agent_names is None:
        agent_names = list(agents_map.keys())

    rdir = Path(results_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    for name in agent_names:
        (rdir / name).mkdir(exist_ok=True)

    instances = load_instances(instances_path)
    instances = [_normalize_instance(inst) for inst in instances]
    completed = _completed_keys(rdir, agent_names)

    plan = []
    for inst in instances:
        for name in agent_names:
            if (name, inst["instance_id"]) in completed:
                continue
            plan.append((name, inst))

    if verbose:
        print(f"Planning {len(plan)} runs "
              f"({len(instances)} instances × {len(agent_names)} agents - "
              f"{len(completed)} already done)")

    t_start = time.time()
    for i, (agent_name, inst) in enumerate(plan):
        agent_cls = agents_map[agent_name]
        try:
            result = run_pair(agent_cls, inst, model, provider,
                              max_turns=max_turns)
        except Exception as e:
            # The agent's solve() catches most exceptions internally, but if
            # something escapes (e.g., import-time errors) we record it here.
            from agents.base import RunResult
            result = RunResult(
                instance_id=inst["instance_id"],
                agent_name=agent_name,
                error=f"{type(e).__name__}: {e}",
                wall_time_s=0.0,
            )

        # Write result
        out_path = _result_path(rdir, agent_name, inst["instance_id"])
        with open(out_path, "w") as f:
            json.dump(asdict(result), f, indent=2)

        if verbose:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (len(plan) - i - 1) if i > 0 else 0
            err = result.error or ""
            print(f"  [{i+1:>4}/{len(plan)}] {agent_name:<20} "
                  f"{inst['instance_id'][:40]:<40} "
                  f"turns={result.turn_count} "
                  f"dur={result.wall_time_s:5.1f}s "
                  f"ETA={eta/60:.1f}m {err[:60]}")

    if verbose:
        print(f"\nDone in {(time.time() - t_start) / 60:.1f}m")
