"""
evaluate.py
===========
Compute primary and secondary metrics from RunResult JSONs.

Primary:
  - PMR: Parameter Match Rate (fraction of instances where the agent's final
    tool call argument for the noisy parameter matches the gold value)
  - CNR: Clarification-Need Recall (fraction where the agent asked a
    clarifying question targeting the expected parameter)

Secondary:
  - Turn count (mean)
  - OverAsk Rate on the clean subset
  - Wrong-Premise Detection Rate (on WRONG_PREMISE subset only)

Pairwise comparison: McNemar's test with continuity correction on PMR,
FEP vs each baseline.
"""

from __future__ import annotations
import json
import math
import re
import argparse
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Synonym dictionary for CNR matching (released as part of the benchmark)
# ---------------------------------------------------------------------------
DEFAULT_SYNONYMS = {
    "location": ["place", "city", "where", "destination"],
    "day": ["date", "when", "time", "weekday"],
    "order_id": ["order", "id", "order number", "which order"],
    "tool_choice": ["which tool", "what kind", "how would you",
                    "which medium", "what format", "by what means",
                    "delivery method", "channel", "how should i"],
}


def _norm(v):
    """Normalise a value for string comparison."""
    if v is None:
        return ""
    return str(v).strip().lower()


def _did_agent_ask_about(transcript: list[dict], target: str,
                         synonyms_db: dict[str, list[str]]) -> bool:
    """Did the agent ask the user a question that references `target` or its
    synonyms?"""
    keys = [target.lower()]
    if "_" in target:
        keys.extend(p.lower() for p in target.split("_") if len(p) >= 2)
    if target in synonyms_db:
        keys.extend(s.lower() for s in synonyms_db[target])
    for m in transcript:
        if m.get("role") != "assistant":
            continue
        if "tool_call" in m:
            continue  # only count text-only assistant turns as clarifications
        content = (m.get("content") or "").lower()
        if not content:
            continue
        if any(k in content for k in keys):
            return True
    return False


def _did_agent_ask_anything(transcript: list[dict]) -> bool:
    """Did the agent emit any text-only assistant turn (a question)?"""
    for m in transcript:
        if m.get("role") != "assistant":
            continue
        if "tool_call" in m:
            continue
        if (m.get("content") or "").strip():
            return True
    return False


def _called_lookup_tool(transcript: list[dict]) -> bool:
    """Did the agent call a status/check/lookup tool at any point?"""
    for m in transcript:
        tc = m.get("tool_call")
        if tc is None:
            continue
        name = (tc.get("name") or "").lower()
        api_part = name.split("__", 1)[1] if "__" in name else name
        if any(kw in api_part for kw in ["check_", "status", "lookup",
                                         "info_", "get_status"]):
            return True
    return False


def _pmr_match(result: dict, instance: dict) -> bool:
    """Does the final tool call's noisy_parameter argument match gold?"""
    fc = result.get("final_tool_call")
    if fc is None:
        return False
    target_param = instance["noisy_parameter"]
    if target_param == "__TOOL__":
        # For UNDERSPECIFIED_TYPE, match on tool name
        return _norm(fc.get("name")) == _norm(instance["gold_tool_name"])
    args = fc.get("arguments") or {}
    return _norm(args.get(target_param)) == _norm(
        instance["noisy_parameter_gold_value"]
    )


# ---------------------------------------------------------------------------
# v1.1 helpers: additional metrics requested by external review
# ---------------------------------------------------------------------------
def _final_tool_name_match(result: dict, instance: dict) -> bool:
    """Tool Choice Accuracy: did the final tool call use the gold tool?
    Independent of argument values."""
    fc = result.get("final_tool_call")
    if fc is None:
        return False
    return _norm(fc.get("name")) == _norm(instance["gold_tool_name"])


def _all_required_args_match(result: dict, instance: dict) -> bool:
    """Final Success: gold tool AND every required gold argument matches.
    Stricter than PMR which only checks the noisy parameter."""
    fc = result.get("final_tool_call")
    if fc is None:
        return False
    if _norm(fc.get("name")) != _norm(instance["gold_tool_name"]):
        return False
    gold_args = instance.get("gold_arguments", {})
    fc_args = fc.get("arguments") or {}
    for k, v in gold_args.items():
        if _norm(fc_args.get(k)) != _norm(v):
            return False
    return True


def _called_tool_before_clarifying(transcript: list[dict]) -> bool:
    """Did the agent's first action involve calling a tool rather than asking?
    Used for Unnecessary Tool Call Rate.
    """
    for m in transcript:
        if m.get("role") != "assistant":
            continue
        if "tool_call" in m:
            return True
        if (m.get("content") or "").strip():
            return False
    return False


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------
def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson CI for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------
def mcnemar_p(b: int, c: int) -> float:
    """McNemar's test with continuity correction.

    b: # of (FEP correct, baseline wrong)
    c: # of (FEP wrong, baseline correct)

    Returns approximate two-sided p-value.
    """
    n = b + c
    if n == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    # Survival of chi-squared(1) = erfc(sqrt(chi2 / 2))
    return math.erfc(math.sqrt(chi2 / 2))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_instances(path: Path) -> dict[str, dict]:
    """Return {instance_id: instance_dict}.

    Handles both NoisyInstance records (have `instance_id`) and CleanSource
    records (have `source_query_id`, no `instance_id`). For the latter,
    synthesise `instance_id = "{source_query_id}__CLEAN"` to match the key
    the runner used when writing result JSONs.
    """
    out = {}
    with open(path) as f:
        for line in f:
            inst = json.loads(line)
            if "instance_id" not in inst:
                qid = inst.get("source_query_id") or inst.get("query_id") or "unknown"
                inst["instance_id"] = f"{qid}__CLEAN"
            if "noise_category" not in inst:
                inst["noise_category"] = "CLEAN"
            out[inst["instance_id"]] = inst
    return out


def load_results(results_dir: Path, agent_name: str) -> dict[str, dict]:
    """Return {instance_id: result_dict} for the given agent."""
    d = results_dir / agent_name
    if not d.exists():
        return {}
    out = {}
    for p in d.glob("*.json"):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        out[r["instance_id"]] = r
    return out


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_metrics(
    instances: dict, results: dict, synonyms: dict,
    is_clean: bool = False,
) -> dict:
    """Compute all metrics for one agent.

    v1.1 metrics (in addition to PMR/CNR/OverAsk/WP-Detect from v1):
      - tca: Tool Choice Accuracy (final tool name match, ignoring args)
      - fsr: Final Success Rate (gold tool AND all gold args match)
      - clarification_precision: of all clarifications asked, fraction
          that targeted the expected parameter (CNR is recall; this is precision)
      - utc_rate: Unnecessary Tool Call Rate — fraction of noisy instances
          where the agent called a tool before any clarification
      - rrac: Recovery Rate After Clarification — of instances where the
          agent asked any clarification, fraction that reached correct PMR
    """
    n = 0
    pmr_hits = 0
    cnr_hits = 0
    tca_hits = 0
    fsr_hits = 0
    asked_any = 0          # denominator for clarification precision
    asked_correct = 0      # numerator for clarification precision
    asked_then_recovered = 0
    utc_count = 0
    overask = 0
    turn_sum = 0
    wp_detect = 0
    wp_total = 0
    per_cat = defaultdict(lambda: {"n": 0, "pmr": 0, "cnr": 0,
                                    "tca": 0, "fsr": 0})

    for inst_id, inst in instances.items():
        result = results.get(inst_id)
        if result is None:
            continue
        n += 1
        cat = inst.get("noise_category", "CLEAN" if is_clean else "?")
        per_cat[cat]["n"] += 1
        transcript = result.get("transcript", [])
        if not is_clean:
            pmr = _pmr_match(result, inst)
            tca = _final_tool_name_match(result, inst)
            fsr = _all_required_args_match(result, inst)
            asked = _did_agent_ask_anything(transcript)
            asked_target = _did_agent_ask_about(
                transcript, inst["expected_clarification_target"], synonyms,
            )
            utc = _called_tool_before_clarifying(transcript)
            if pmr:
                pmr_hits += 1
                per_cat[cat]["pmr"] += 1
            if tca:
                tca_hits += 1
                per_cat[cat]["tca"] += 1
            if fsr:
                fsr_hits += 1
                per_cat[cat]["fsr"] += 1
            if asked_target:
                cnr_hits += 1
                per_cat[cat]["cnr"] += 1
            if asked:
                asked_any += 1
                if asked_target:
                    asked_correct += 1
                if pmr:
                    asked_then_recovered += 1
            if utc:
                utc_count += 1
            if cat == "WRONG_PREMISE":
                wp_total += 1
                if _called_lookup_tool(transcript):
                    wp_detect += 1
        else:
            if _did_agent_ask_anything(transcript):
                overask += 1
        turn_sum += result.get("turn_count", 0)

    metrics = {
        "n": n,
        "mean_turns": turn_sum / n if n > 0 else 0,
    }
    if not is_clean:
        metrics.update({
            "pmr": pmr_hits / n if n > 0 else 0,
            "pmr_ci": wilson_ci(pmr_hits / n if n > 0 else 0, n),
            "cnr": cnr_hits / n if n > 0 else 0,
            "cnr_ci": wilson_ci(cnr_hits / n if n > 0 else 0, n),
            # v1.1 additions
            "tca": tca_hits / n if n > 0 else 0,
            "fsr": fsr_hits / n if n > 0 else 0,
            "clarification_precision": (asked_correct / asked_any
                                         if asked_any > 0 else 0),
            "utc_rate": utc_count / n if n > 0 else 0,
            "rrac": (asked_then_recovered / asked_any
                     if asked_any > 0 else 0),
            "n_asked": asked_any,
            "per_category": {
                cat: {
                    "n": d["n"],
                    "pmr": d["pmr"] / d["n"] if d["n"] > 0 else 0,
                    "cnr": d["cnr"] / d["n"] if d["n"] > 0 else 0,
                    "tca": d["tca"] / d["n"] if d["n"] > 0 else 0,
                    "fsr": d["fsr"] / d["n"] if d["n"] > 0 else 0,
                }
                for cat, d in per_cat.items()
            },
            "wp_detect_rate": wp_detect / wp_total if wp_total > 0 else 0,
            "wp_n": wp_total,
        })
    else:
        metrics["overask_rate"] = overask / n if n > 0 else 0
    return metrics


def pairwise_mcnemar(
    instances: dict, results_a: dict, results_b: dict,
) -> dict:
    """Compute paired PMR comparison between agents a and b.

    Returns counts of {both_correct, a_only, b_only, both_wrong} plus McNemar p.
    """
    bb = aa = both = neither = 0
    for inst_id, inst in instances.items():
        ra = results_a.get(inst_id)
        rb = results_b.get(inst_id)
        if ra is None or rb is None:
            continue
        a_ok = _pmr_match(ra, inst)
        b_ok = _pmr_match(rb, inst)
        if a_ok and b_ok:
            both += 1
        elif a_ok and not b_ok:
            aa += 1  # FEP wins
        elif not a_ok and b_ok:
            bb += 1  # baseline wins
        else:
            neither += 1
    return {
        "a_only": aa, "b_only": bb,
        "both_correct": both, "both_wrong": neither,
        "n_paired": aa + bb + both + neither,
        "mcnemar_p": mcnemar_p(aa, bb),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(
    instances_noisy: dict, instances_clean: dict,
    results_dir: Path, agent_names: list[str], synonyms: dict,
):
    print("=" * 78)
    print(f"AmbiguousToolBench evaluation")
    print(f"  Noisy instances: {len(instances_noisy)}")
    print(f"  Clean instances: {len(instances_clean)}")
    print(f"  Agents:          {agent_names}")
    print("=" * 78)

    # Per-agent table
    all_metrics = {}
    for agent in agent_names:
        results = load_results(results_dir, agent)
        m_noisy = compute_metrics(instances_noisy, results, synonyms, is_clean=False)
        m_clean = compute_metrics(instances_clean, results, synonyms, is_clean=True)
        all_metrics[agent] = {"noisy": m_noisy, "clean": m_clean}
        print(f"\n--- {agent} ---")
        print(f"  n={m_noisy['n']} noisy, {m_clean['n']} clean")
        print(f"  PMR: {m_noisy['pmr']:.3f}  "
              f"95% CI [{m_noisy['pmr_ci'][0]:.3f}, {m_noisy['pmr_ci'][1]:.3f}]")
        print(f"  CNR: {m_noisy['cnr']:.3f}  "
              f"95% CI [{m_noisy['cnr_ci'][0]:.3f}, {m_noisy['cnr_ci'][1]:.3f}]")
        # v1.1 metrics
        print(f"  Tool Choice Accuracy: {m_noisy['tca']:.3f}")
        print(f"  Final Success Rate:   {m_noisy['fsr']:.3f}")
        print(f"  Clarif. Precision:    {m_noisy['clarification_precision']:.3f}"
              f"  (asked: {m_noisy['n_asked']})")
        print(f"  Unnecessary ToolCall: {m_noisy['utc_rate']:.3f}")
        print(f"  Recovery|Asked (RRAC):{m_noisy['rrac']:.3f}")
        print(f"  Mean turns (noisy):   {m_noisy['mean_turns']:.2f}")
        print(f"  OverAsk on clean:     {m_clean['overask_rate']:.3f}")
        if m_noisy["wp_n"] > 0:
            print(f"  Wrong-Premise detection: "
                  f"{m_noisy['wp_detect_rate']:.3f} (n={m_noisy['wp_n']})")
        print(f"  Per-category breakdown:")
        for cat in sorted(m_noisy["per_category"]):
            d = m_noisy["per_category"][cat]
            print(f"    {cat:<22} n={d['n']:>3} "
                  f"PMR={d['pmr']:.3f}  TCA={d['tca']:.3f}  "
                  f"FSR={d['fsr']:.3f}  CNR={d['cnr']:.3f}")

    # Pairwise McNemar (StructuralFEP vs others)
    print("\n" + "=" * 78)
    print("Pairwise PMR comparison (StructuralFEP vs each baseline)")
    print("=" * 78)
    fep_results = load_results(results_dir, "StructuralFEP")
    if not fep_results:
        print("(No StructuralFEP results loaded — skipping pairwise comparison)")
    else:
        for agent in agent_names:
            if agent == "StructuralFEP":
                continue
            baseline_results = load_results(results_dir, agent)
            if not baseline_results:
                continue
            comp = pairwise_mcnemar(instances_noisy, fep_results, baseline_results)
            print(f"\n  FEP vs {agent}:")
            print(f"    FEP only correct:      {comp['a_only']}")
            print(f"    {agent[:24]:<24} only:  {comp['b_only']}")
            print(f"    Both correct:          {comp['both_correct']}")
            print(f"    Both wrong:            {comp['both_wrong']}")
            print(f"    McNemar p:             {comp['mcnemar_p']:.4f}")
    return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="data/atb_v1.jsonl")
    ap.add_argument("--clean-instances", default="data/atb_clean_v1.jsonl")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--agents", nargs="+",
                    default=["PlainToolCalling", "ReAct", "HeuristicClarify",
                             "PromptedClarify", "StructuralFEP"])
    ap.add_argument("--synonyms", default=None,
                    help="Path to clarification_synonyms.json")
    ap.add_argument("--output", default=None,
                    help="Optional CSV output path")
    args = ap.parse_args()

    instances_noisy = load_instances(Path(args.instances))
    instances_clean = (load_instances(Path(args.clean_instances))
                       if Path(args.clean_instances).exists() else {})
    syn = DEFAULT_SYNONYMS
    if args.synonyms and Path(args.synonyms).exists():
        with open(args.synonyms) as f:
            syn = {**DEFAULT_SYNONYMS, **json.load(f)}

    metrics = print_report(
        instances_noisy, instances_clean,
        Path(args.results_dir), args.agents, syn,
    )

    if args.output:
        # Flatten to CSV — v1.1 includes additional metrics
        rows = []
        for agent, m in metrics.items():
            noisy = m["noisy"]
            clean = m["clean"]
            rows.append({
                "agent": agent,
                "n": noisy["n"],
                "pmr": noisy["pmr"],
                "tca": noisy["tca"],
                "fsr": noisy["fsr"],
                "cnr": noisy["cnr"],
                "clarification_precision": noisy["clarification_precision"],
                "utc_rate": noisy["utc_rate"],
                "rrac": noisy["rrac"],
                "n_asked": noisy["n_asked"],
                "mean_turns": noisy["mean_turns"],
                "overask_rate": clean.get("overask_rate", None),
                "wp_detect_rate": noisy.get("wp_detect_rate", None),
                "wp_n": noisy.get("wp_n", None),
            })
        import csv
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
