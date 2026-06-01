# Benchmark Specification

This document is the formal specification of AmbiguousToolBench v1.1.1. The README gives the headline; this fills in the details a reviewer or future contributor would need.

## Goal

Measure whether tool-using LLM agents recognise underspecified user queries and ask before acting, vs. confidently calling a tool with a plausible-looking but wrong value.

## Source corpus

ATB is built from the *simple* split of the [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) v3 at commit `70b6a4a`.

Selection criteria for ATB inclusion:
- Single-turn (one user query → one tool call)
- Single-tool (one function in the registry, no selection ambiguity)
- Has at least one required string, integer, or float parameter that admits a deterministic noise transformation

Of BFCL v3 *simple* (400 instances), 187 met the criteria after applying the two transformations.

## Noise taxonomy

### MISSING_REQUIRED (40 instances)

A required argument is replaced in the user query by a type-appropriate placeholder.

| Argument type | Placeholder |
|---|---|
| Person, organisation (string) | *someone* |
| Numeric (int, float) | *some* |
| Other strings | *something* |

The tool registry, parameter schema, and gold call are unchanged. Only the user query string is altered.

**Example.**
- Original: *Predict the future value of a $5000 investment with 5% return for 3 years*
- Noisy: *Predict the future value of a $some investment with 5% return for 3 years*
- Gold value to recover: `5000`

### AMBIGUOUS_REFERENCE (40 instances)

A named entity in the query is replaced by a context-appropriate pronoun.

| Entity type | Pronoun |
|---|---|
| Person, organisation | *them*, *that person* |
| Location | *there*, *that place* |
| Object, item | *that one* |

**Example.**
- Original: *Give me the schedule of Manchester United for the next 6 games in Premier League*
- Noisy: *Give me the schedule of them for the next 6 games in Premier League*
- Gold value to recover: `Manchester United`

### WRONG_PREMISE (7 instances, pilot)

A pilot category included for transparency but excluded from headline analysis. The query asserts something false about the world (e.g., a non-existent entity) that the agent should question rather than tool-call against. The oracle simulator design makes premise verification trivially recoverable via clarification, so this category does not separate agent designs. Reserved for future expansion with a richer simulator.

### Hand-fixes

15 AMBIGUOUS_REFERENCE instances received manual fixes where the automated pronoun substitution produced an ungrammatical or implausible result (e.g., *them* substituted for an inanimate object). Both the original automated outputs and the hand-fixes are released alongside the dataset; the commit log documents each fix.

## Construction is deterministic

Given:
- BFCL v3 at the pinned commit
- Random seed 42 (for sampling 100 clean instances)
- The 15 documented hand-fixes

A fresh checkout of this repo reproduces `data/atb_v1.jsonl` and `data/atb_clean_v1.jsonl` byte-for-byte. The construction script lives in `src/construct_dataset.py`.

## Agents

Five reference agents are included in `src/agents/`. They share a maximum-turns budget (8) and the oracle user simulator, differing only in system prompt and (for `HeuristicClarify` and `StructuralFEP`) in control logic outside the language model. See `src/agents/base.py` for the common contract.

## Metrics

Seven quantities per agent. Five on noisy inputs, two on clean inputs.

### Noisy-input metrics

**Parameter Match Rate (PMR).** Headline metric. Fraction of noisy instances where:
- the agent's final tool call uses the gold tool, AND
- the agent's final tool call for the noisy parameter exactly matches the gold value (with type coercion for numerics).

**Final Success Rate (FSR).** Stricter than PMR: requires *every* required parameter (not just the noisy one) to match gold. FSR ≤ PMR.

**Tool Choice Accuracy (TCA).** Just the gold tool name, ignoring arguments. This is what current public benchmarks report.

**Clarification Precision (CP).** Among instances where the agent asked at least one clarifying question, the fraction where the question targeted the gold `expected_clarification_target`. Undefined when no instance triggered an ask.

**Recovery Rate Among Clarified (RRAC).** Among ask-instances, the fraction where the agent then succeeded in the PMR sense. Measures whether the agent can productively use a clarification answer.

### Clean-input metrics

**OverAsk on Clean.** Fraction of 100 clean instances on which the agent asked a clarifying question. High OverAsk = cautious-to-the-point-of-annoying.

**Unnecessary Tool Call rate (UTC).** Fraction of noisy instances where the agent issued a tool call before the noise was resolved (either by asking or by inferring from context).

### Statistics

- 95% Wilson intervals for PMR (and other proportions)
- McNemar test on per-instance agreement for pairwise comparisons (paired binary outcomes; correct asymptotic distribution)

## Versioning

This document specifies ATB v1.1.1. Any changes to the noise taxonomy, dataset, or metrics bump the version. Reproducibility is the contract: the same version of ATB on the same BFCL commit should yield the same dataset byte-for-byte. Results across versions are not directly comparable.

## Licensing

- Code: MIT
- Dataset: CC-BY 4.0 (derivative of BFCL, also Apache 2.0)
- Paper: CC-BY 4.0
