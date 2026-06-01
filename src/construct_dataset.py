"""
construct_dataset.py
====================
Build AmbiguousToolBench v1 from BFCL v3 simple.

Source: https://github.com/ShishirPatil/gorilla/tree/master/berkeley-function-call-leaderboard/data
  - BFCL_v3_simple.json                  : questions + tool registries
  - possible_answer/BFCL_v3_simple.json  : gold arguments per question id

Pipeline:
  1. Load BFCL questions (JSONL) and the matching answer file (JSONL),
     keyed by 'id'.
  2. For each question, normalise into a CleanSource:
       - query  = question[0][0]['content']
       - tool_registry = function (single tool in 'simple' category)
       - gold_tool_name = function[0]['name']
       - gold_arguments = answer['ground_truth'][param_name][0]  (BFCL stores
         each gold value as a list of accepted alternatives)
  3. For each source, compute eligibility per noise category.
  4. Assign noise categories round-robin, balanced toward ~40 per category.
  5. Apply the transform; collect NoisyInstances.
  6. Also pass through 100 unmodified sources as the CLEAN held-out set.
  7. Write JSONL to data/atb_v1.jsonl and data/atb_clean_v1.jsonl.

No LLM is called. Construction is deterministic given a seed.

Usage:
    python construct_dataset.py \\
        --questions-file BFCL_v3_simple.json \\
        --answers-file possible_answer/BFCL_v3_simple.json \\
        --output-dir data \\
        --seed 0 \\
        --target-per-category 40 \\
        --clean-target 100
"""

from __future__ import annotations
import argparse
import json
import random
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from noise_transforms import (
    CleanSource, NoisyInstance, ELIGIBILITY, apply_noise,
)


# ---------------------------------------------------------------------------
# BFCL → CleanSource
# ---------------------------------------------------------------------------
def _normalise_param_schema(bfcl_params: dict) -> dict:
    """BFCL's parameter schemas use `type: 'dict'` for the outer object and
    keep parameter properties under `properties`. The properties themselves
    are mostly OpenAI-compatible already, with a few quirks:
      - some types are 'tuple' or 'any' instead of 'array' / 'string'
      - some 'array' types specify `items` as a dict
    For our purposes (eligibility checks + noise transforms), we only need
    `type`, optional `enum`, optional `description`. Normalise lightly.
    """
    out = {}
    for pname, pspec in bfcl_params.get("properties", {}).items():
        if not isinstance(pspec, dict):
            continue
        ptype = pspec.get("type", "string")
        type_map = {
            "string": "string", "integer": "integer", "number": "number",
            "float": "number", "boolean": "boolean", "array": "array",
            "dict": "object", "object": "object", "tuple": "array",
            "any": "string",
        }
        jtype = type_map.get(ptype, "string")
        entry = {"type": jtype}
        if pspec.get("description"):
            entry["description"] = pspec["description"]
        if pspec.get("enum"):
            entry["enum"] = pspec["enum"]
        out[pname] = entry
    return out


def _build_tool_registry(bfcl_function_list: list[dict]) -> list[dict]:
    """Convert BFCL `function` field into OpenAI-format tool registry."""
    registry = []
    for fn in bfcl_function_list:
        if not isinstance(fn, dict):
            continue
        fn_name = fn.get("name", "")
        if not fn_name:
            continue
        # Sanitise name: keep alphanumerics, underscore, period; replace others
        safe_name = "".join(
            c if (c.isalnum() or c in ("_", ".")) else "_" for c in fn_name
        )
        params = fn.get("parameters", {})
        props = _normalise_param_schema(params)
        required = list(params.get("required", []))
        registry.append({
            "type": "function",
            "function": {
                "name": safe_name,
                "description": fn.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        })
    return registry


def _extract_gold_arguments(answer_record: dict) -> dict:
    """BFCL `possible_answer` stores ground_truth as a list of dicts. Each
    outer dict has the gold tool name as its single key, and the value is
    a dict mapping each parameter name to a list of accepted alternative
    values. We take the first accepted value for each parameter.

    Real format:
      {
        "id": "simple_0",
        "ground_truth": [{
            "calculate_triangle_area": {
                "base": [10],
                "height": [5],
                "unit": ["units", ""]
            }
        }]
      }
    """
    gt = answer_record.get("ground_truth", [])
    if not gt or not isinstance(gt, list):
        return {}
    outer = gt[0]
    if not isinstance(outer, dict) or not outer:
        return {}
    # Unwrap the tool-name layer. There should be exactly one key (the gold
    # tool name); take the inner dict whatever the key.
    arg_dict = next(iter(outer.values()))
    if not isinstance(arg_dict, dict):
        return {}
    out = {}
    for pname, alts in arg_dict.items():
        if isinstance(alts, list) and alts:
            v = alts[0]
            # Empty-string default means optional param — skip
            if v != "" or len(alts) == 1:
                out[pname] = v
        else:
            out[pname] = alts
    return out


def parse_bfcl_record(question: dict, answer: dict | None) -> CleanSource | None:
    """Combine a BFCL question record and its answer into a CleanSource.

    Returns None if the record is malformed, has no gold answer, or has
    gold parameter values that aren't grounded in the query text.
    """
    qid = question.get("id")
    question_field = question.get("question", [])
    if not qid or not question_field:
        return None
    try:
        first_turn = question_field[0]
        if isinstance(first_turn, list) and first_turn:
            user_msg = first_turn[0].get("content", "")
        elif isinstance(first_turn, dict):
            user_msg = first_turn.get("content", "")
        else:
            return None
    except (IndexError, AttributeError, TypeError):
        return None
    if not user_msg:
        return None

    functions = question.get("function", [])
    if not functions:
        return None
    registry = _build_tool_registry(functions)
    if not registry:
        return None
    gold_tool_name = registry[0]["function"]["name"]
    gold_schema = registry[0]["function"]["parameters"]["properties"]

    if answer is None:
        return None
    gold_args = _extract_gold_arguments(answer)
    if not gold_args:
        return None

    # Restrict to required params only — they're what noise transforms target.
    # Optional params with defaults are dropped from gold_args because the
    # agent isn't required to produce them.
    required = registry[0]["function"]["parameters"].get("required", [])
    gold_args = {p: v for p, v in gold_args.items() if p in required}
    if not gold_args:
        return None

    # Verify at least one gold value is grounded in the query text. If none
    # are grounded, no noise transform can meaningfully target a parameter.
    if not _at_least_one_grounded(user_msg, gold_args):
        return None

    return CleanSource(
        source_query_id=str(qid),
        query=user_msg,
        tool_registry=registry,
        gold_tool_name=gold_tool_name,
        gold_arguments=gold_args,
        parameter_schema=gold_schema,
    )


def _at_least_one_grounded(query: str, gold_args: dict) -> bool:
    """A parameter is 'grounded' if its gold value appears in the query
    text. Numeric values use word-boundary matching; strings use substring.
    """
    ql = query.lower()
    for v in gold_args.values():
        if v in (None, "", []):
            continue
        s = str(v)
        if not s:
            continue
        if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
            if re.search(r"(?<!\d)" + re.escape(s) + r"(?!\d)", query):
                return True
        else:
            if len(s) >= 2 and s.lower() in ql:
                return True
    return False


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_jsonl(path: Path) -> list[dict]:
    """BFCL files are line-delimited JSON, one record per line."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARN: bad JSON line in {path}: {e}", file=sys.stderr)
    return out


def load_sources(questions_path: Path, answers_path: Path) -> list[CleanSource]:
    questions = _load_jsonl(questions_path)
    answers = _load_jsonl(answers_path)
    by_id = {a["id"]: a for a in answers if "id" in a}
    print(f"  loaded {len(questions)} questions, {len(answers)} answers")

    sources = []
    drops = Counter()
    for q in questions:
        qid = q.get("id")
        a = by_id.get(qid) if qid else None
        if a is None:
            drops["no_answer"] += 1
            continue
        s = parse_bfcl_record(q, a)
        if s is None:
            drops["unparseable_or_ungrounded"] += 1
            continue
        sources.append(s)
    print(f"  parsed {len(sources)} CleanSources "
          f"(dropped {sum(drops.values())}: {dict(drops)})")
    return sources


# ---------------------------------------------------------------------------
# Round-robin category assignment
# ---------------------------------------------------------------------------
def assign_noise_categories(
    sources: list[CleanSource], target_per_cat: int, seed: int,
) -> dict[str, list[tuple[CleanSource, str]]]:
    categories = list(ELIGIBILITY.keys())
    quotas = {c: target_per_cat for c in categories}
    assigned: dict[str, list] = {c: [] for c in categories}

    rng = random.Random(seed)
    shuffled = list(sources)
    rng.shuffle(shuffled)

    for s in shuffled:
        eligible_cats = []
        for c in categories:
            if quotas[c] <= 0:
                continue
            params = ELIGIBILITY[c](s)
            if params:
                eligible_cats.append(c)
        if not eligible_cats:
            continue
        eligible_cats.sort(key=lambda c: (-quotas[c], c))
        chosen = eligible_cats[0]
        assigned[chosen].append((s, chosen))
        quotas[chosen] -= 1
        if all(q == 0 for q in quotas.values()):
            break
    return assigned


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_dataset(
    sources: list[CleanSource], target_per_cat: int, clean_target: int,
    seed: int,
) -> tuple[list[NoisyInstance], list[CleanSource], dict]:
    assigned = assign_noise_categories(sources, target_per_cat, seed)

    noisy = []
    failures = Counter()
    for cat, items in assigned.items():
        for s, c in items:
            try:
                noisy.append(apply_noise(s, c, seed=seed))
            except Exception as e:
                failures[f"{cat}: {type(e).__name__}"] += 1

    used_qids = {n.source_query_id for n in noisy}
    rng = random.Random(seed + 1)
    unused = [s for s in sources if s.source_query_id not in used_qids]
    rng.shuffle(unused)
    clean = unused[:clean_target]

    stats = {
        "n_sources_loaded": len(sources),
        "n_noisy": len(noisy),
        "n_clean": len(clean),
        "per_category_counts": dict(Counter(n.noise_category for n in noisy)),
        "transform_failures": dict(failures),
    }
    return noisy, clean, stats


def write_jsonl(items: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(asdict(item)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-file", type=Path, required=True,
                    help="Path to BFCL_v3_simple.json (JSONL)")
    ap.add_argument("--answers-file", type=Path, required=True,
                    help="Path to possible_answer/BFCL_v3_simple.json (JSONL)")
    ap.add_argument("--output-dir", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-per-category", type=int, default=40)
    ap.add_argument("--clean-target", type=int, default=100)
    args = ap.parse_args()

    sources = load_sources(args.questions_file, args.answers_file)
    if not sources:
        print("FATAL: no sources parsed.", file=sys.stderr)
        sys.exit(1)

    noisy, clean, stats = build_dataset(
        sources, args.target_per_category, args.clean_target, args.seed,
    )
    write_jsonl(noisy, args.output_dir / "atb_v1.jsonl")
    write_jsonl(clean, args.output_dir / "atb_clean_v1.jsonl")

    print("\n=== Construction stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nWrote:")
    print(f"  {args.output_dir / 'atb_v1.jsonl'}")
    print(f"  {args.output_dir / 'atb_clean_v1.jsonl'}")


if __name__ == "__main__":
    main()
