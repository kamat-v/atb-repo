"""
noise_transforms.py
===================
Five deterministic noise transformations for AmbiguousToolBench.

Each transform takes a clean (query, tool_call) source and produces a noisy
variant with recoverable ground truth. Transforms are pure functions of
(source, seed) — same inputs always produce same outputs.

See benchmark_spec.md §3 for the taxonomy and worked examples.
"""

from __future__ import annotations
import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Source / output data model
# ---------------------------------------------------------------------------
@dataclass
class CleanSource:
    """A clean (query, tool_call) pair from the source corpus."""
    source_query_id: str
    query: str                                  # the natural-language user instruction
    tool_registry: list[dict]                   # OpenAI function-call format
    gold_tool_name: str
    gold_arguments: dict[str, Any]
    parameter_schema: dict[str, dict] = field(default_factory=dict)
    # parameter_schema[param_name] = {"type": "string"|"number"|..., "enum": [...]?}


@dataclass
class NoisyInstance:
    """The output of applying a noise transform."""
    instance_id: str
    source_query_id: str
    noise_category: str
    noisy_query: str
    original_query: str
    tool_registry: list[dict]
    gold_tool_name: str
    gold_arguments: dict[str, Any]
    parameter_schema: dict[str, dict]
    noisy_parameter: str
    noisy_parameter_gold_value: Any
    expected_clarification_target: str


# ---------------------------------------------------------------------------
# Eligibility checks
# ---------------------------------------------------------------------------
def eligible_for_missing_required(s: CleanSource) -> list[str]:
    """Return list of required-parameter names whose value appears as a literal
    in the query text (so we can locate and remove it)."""
    out = []
    for pname, pval in s.gold_arguments.items():
        if pval in (None, "", []):
            continue
        pstr = str(pval)
        if len(pstr) < 2:
            continue  # too short to safely substring-match
        if pstr.lower() in s.query.lower():
            out.append(pname)
    return out


def eligible_for_ambiguous_reference(s: CleanSource) -> list[str]:
    """Parameters that look like specific IDs/names — strings of length >=3
    that appear in the query and have either uppercase chars, digits, or
    look like a proper noun."""
    out = []
    for pname, pval in s.gold_arguments.items():
        if not isinstance(pval, str) or len(pval) < 3:
            continue
        if pval.lower() not in s.query.lower():
            continue
        # Must look like an identifier: contains digit, contains uppercase,
        # or is a capitalised proper noun
        has_digit = any(c.isdigit() for c in pval)
        has_upper = any(c.isupper() for c in pval[1:])
        is_proper = pval[0].isupper()
        if has_digit or has_upper or is_proper:
            out.append(pname)
    return out


def eligible_for_wrong_premise(s: CleanSource) -> list[str]:
    """Parameters whose schema has an enum (so we can pick a wrong-but-valid
    alternative) AND the registry contains a tool with 'status' or 'check'
    or 'get' in its name suitable for verification."""
    has_lookup_tool = any(
        any(kw in t.get("function", {}).get("name", "").lower()
            for kw in ["status", "check", "get_", "lookup", "info"])
        for t in s.tool_registry
    )
    if not has_lookup_tool:
        return []
    out = []
    for pname, pval in s.gold_arguments.items():
        schema = s.parameter_schema.get(pname, {})
        enum = schema.get("enum")
        if enum and len(enum) >= 2 and pval in enum:
            out.append(pname)
    return out


def eligible_for_underspecified_type(s: CleanSource) -> list[str]:
    """Tools whose name contains a type-specifier prefix and there's at least
    one sibling tool sharing that prefix (so the registry has alternatives
    the agent could plausibly confuse for the gold tool).

    Handles two name styles:
      - Bare prefix: "send_email", "get_status"
      - Tool__API style: "Messaging__send_email" (StableToolBench format)

    Returns ['__TOOL__'] as a sentinel (this category noises the tool
    choice, not a parameter)."""
    gold_name = s.gold_tool_name.lower()
    prefixes = ["send_", "get_", "fetch_", "post_", "save_", "export_"]

    # Strip Tool__ prefix if present (StableToolBench style)
    if "__" in gold_name:
        api_part = gold_name.split("__", 1)[1]
    else:
        api_part = gold_name

    has_prefix = None
    for p in prefixes:
        if api_part.startswith(p):
            has_prefix = p
            break
    if not has_prefix:
        return []

    # Count sibling tools with the same prefix (after stripping Tool__)
    siblings = 0
    for t in s.tool_registry:
        tname = t.get("function", {}).get("name", "").lower()
        if tname == gold_name:
            continue
        sibling_api = tname.split("__", 1)[1] if "__" in tname else tname
        if sibling_api.startswith(has_prefix):
            siblings += 1
    if siblings >= 1:
        return ["__TOOL__"]
    return []


def eligible_for_volume_scope(s: CleanSource) -> list[str]:
    """Parameters that reference a single instance whose pluralisation
    would be ambiguous. We require the parameter name to map to a known
    noun root in VOLUME_PLURALISATIONS — otherwise the transform produces
    a grammatically valid but semantically odd output that doesn't actually
    exhibit volume/scope ambiguity.
    """
    out = []
    for pname, pval in s.gold_arguments.items():
        if not isinstance(pval, str) or len(pval) < 2:
            continue
        if pval.lower() not in s.query.lower():
            continue
        # Heuristic: parameter name must suggest a single countable noun
        # whose plural form is in our vocabulary
        pname_lower = pname.lower()
        if not any(root in pname_lower for root in VOLUME_PLURALISATIONS):
            continue
        out.append(pname)
    return out


# Map category to eligibility checker
ELIGIBILITY = {
    "MISSING_REQUIRED": eligible_for_missing_required,
    "AMBIGUOUS_REFERENCE": eligible_for_ambiguous_reference,
    "WRONG_PREMISE": eligible_for_wrong_premise,
    # UNDERSPECIFIED_TYPE: removed from v1.1. BFCL simple's tool registries
    # typically contain one tool, so no sibling tools share a verb prefix.
    # n=0 instances in v1. The function `eligible_for_underspecified_type`
    # is retained for v1.2 / alternative source corpora.
    # VOLUME_SCOPE: removed from v1.1. The regex-based transform produced
    # many broken instances on BFCL queries (dangling prepositions, empty
    # quotes, double-prepositions). 8 of 32 v1 instances were obviously
    # malformed; further audit showed most "OK" instances also had
    # grammatical artefacts. v1.2 will require grammar-preserving template
    # generation. The function `eligible_for_volume_scope` is retained.
}


# ---------------------------------------------------------------------------
# Replacement vocabularies
# ---------------------------------------------------------------------------
MISSING_REPLACEMENTS = {
    "string_location": "somewhere",
    "string_person": "someone",
    "string_date": "at some point",
    "string_generic": "something",
    "number": "some",
}


AMBIGUOUS_REPLACEMENTS = {
    # Category-specific replacements. Per reviewer (v1.1): avoid unnatural
    # generic placeholders like "the thing" or bare "someone". Each list is
    # ordered roughly from most-natural to least-natural; we sample
    # deterministically from the list.
    "id": ["it", "this one", "that one"],
    "person": ["them", "this person", "that person"],
    "location": ["there", "that place"],
    "date": ["then", "at that time", "on that date"],
    "object": ["that one", "this one", "it"],
    "title": ["that one", "the one I mentioned"],
    "generic": ["that", "the one", "it"],  # last-resort fallback
}


VOLUME_PLURALISATIONS = {
    "order": "orders",
    "flight": "flights",
    "item": "items",
    "product": "products",
    "message": "messages",
    "file": "files",
    "document": "documents",
    "user": "users",
    "account": "accounts",
    "task": "tasks",
    "company": "companies",
    "restaurant": "restaurants",
    "movie": "movies",
    "city": "cities",
    "country": "countries",
    "book": "books",
    "stock": "stocks",
    "actor": "actors",
    "song": "songs",
    "place": "places",
}


# ---------------------------------------------------------------------------
# Transform implementations
# ---------------------------------------------------------------------------
def _classify_param_for_missing(pname: str, pval: Any) -> str:
    """Pick a replacement category for MISSING_REQUIRED."""
    pname_lower = pname.lower()
    if isinstance(pval, (int, float)):
        return "number"
    if any(kw in pname_lower for kw in ["city", "location", "place", "country", "address"]):
        return "string_location"
    if any(kw in pname_lower for kw in ["name", "person", "user", "customer", "client"]):
        return "string_person"
    if any(kw in pname_lower for kw in ["date", "time", "day", "month", "year"]):
        return "string_date"
    return "string_generic"


def _classify_param_for_ambig(pname: str) -> str:
    """Map parameter name to a replacement-vocab category.

    v1.1 (post-review): order matters, AND we use word-boundary matching to
    avoid substring false positives (e.g. 'actor' should not match 'to' in
    'location'). Names are split on _, camelCase, and digits.
    """
    import re
    # Tokenize parameter name into lowercase word components
    # e.g. "company_name" → {"company", "name"}
    #      "recipeName"   → {"recipe", "name"}
    #      "start_x"      → {"start", "x"}
    tokens = set(re.findall(r"[a-z]+", re.sub(r"([A-Z])", r"_\1", pname).lower()))

    def has_any(*words):
        return any(w in tokens for w in words)

    # 1. IDs and codes
    if has_any("id", "docket", "number", "code", "ssn"):
        return "id"
    # Special suffix cases
    pl = pname.lower()
    if pl.endswith("_id") or pl.endswith("id") and len(pl) <= 4:
        return "id"
    # 2. People — explicit person nouns (BEFORE generic 'name')
    if has_any("person", "user", "customer", "actor", "artist", "author",
                "speaker", "host", "guest", "patient", "lawyer", "papal",
                "pope", "priest", "professor", "doctor"):
        return "person"
    # 3. Player can be either (sports vs games) — bias to person
    if has_any("player") and not has_any("game", "media"):
        return "person"
    # 4. Organizations / venues / brands
    if has_any("company", "organization", "org", "manufacturer", "brand",
                "hotel", "museum", "venue", "restaurant", "shop", "store",
                "team", "league", "bank", "school", "university", "database"):
        return "object"
    # 5. Locations
    if has_any("city", "location", "place", "loc", "country", "destination",
                "origin", "court", "address", "region", "area", "site",
                "departure", "arrival", "venue", "from", "to"):
        return "location"
    # 6. Dates/times
    if has_any("date", "time", "day", "year", "season", "month", "week",
                "hour", "minute"):
        return "date"
    # 7. Titles/works
    if has_any("title", "song", "movie", "film", "book", "game", "album",
                "track", "recipe"):
        return "title"
    # 8. Items/products
    if has_any("item", "product", "object", "ingredient", "dish", "food",
                "instrument", "drink", "vehicle"):
        return "object"
    # 9. Generic "name" fallback: object (safer than person on unknown nouns)
    if "name" in tokens:
        return "object"
    return "generic"


def _case_insensitive_replace(text: str, old: str, new: str) -> str:
    """Replace the first occurrence of `old` in `text`, case-insensitively,
    preserving the rest of the string.

    For numeric `old` (pure digits, optionally with decimal point/sign), use
    word-boundary matching so that "5" doesn't match inside "15". For string
    values, fall back to substring matching, which is what we want for proper
    nouns (e.g. matching "Paris" inside "Parisian" is rare).
    """
    old_str = str(old)
    if not old_str:
        return text
    # Numeric? Use word-boundary regex.
    if re.fullmatch(r'-?\d+(?:\.\d+)?', old_str):
        pattern = re.compile(r'(?<!\d)' + re.escape(old_str) + r'(?!\d)')
        m = pattern.search(text)
        if m is None:
            return text
        return text[:m.start()] + new + text[m.end():]
    # String? Plain case-insensitive substring.
    idx = text.lower().find(old_str.lower())
    if idx < 0:
        return text
    return text[:idx] + new + text[idx + len(old_str):]


def _missing_required(s: CleanSource, param: str, rng: random.Random) -> NoisyInstance:
    pval = s.gold_arguments[param]
    cat = _classify_param_for_missing(param, pval)
    replacement = MISSING_REPLACEMENTS[cat]
    noisy_query = _case_insensitive_replace(s.query, str(pval), replacement)
    return NoisyInstance(
        instance_id=f"{s.source_query_id}__MISSING_REQUIRED__{param}",
        source_query_id=s.source_query_id,
        noise_category="MISSING_REQUIRED",
        noisy_query=noisy_query,
        original_query=s.query,
        tool_registry=s.tool_registry,
        gold_tool_name=s.gold_tool_name,
        gold_arguments=s.gold_arguments,
        parameter_schema=s.parameter_schema,
        noisy_parameter=param,
        noisy_parameter_gold_value=pval,
        expected_clarification_target=param,
    )


def _ambiguous_reference(s: CleanSource, param: str, rng: random.Random) -> NoisyInstance:
    pval = s.gold_arguments[param]
    cat = _classify_param_for_ambig(param)
    options = AMBIGUOUS_REPLACEMENTS[cat]
    replacement = rng.choice(options)
    # Detect a preceding noun like "order W12345" -> "it"; need to absorb the noun
    # so we don't get "order it". We look for "<noun> <pval>" where noun is in
    # a known list of identifier-noun anchors.
    anchors = ["order", "flight", "ticket", "account", "user",
               "customer", "item", "product", "file", "document",
               "task", "ticket", "booking", "reservation"]
    noisy_query = s.query
    consumed_anchor = False
    for anchor in anchors:
        # Match "anchor <pval>" optionally with " #" between
        pattern = re.compile(
            r'\b' + re.escape(anchor) + r'\s+#?' + re.escape(str(pval)) + r'\b',
            re.IGNORECASE,
        )
        if pattern.search(noisy_query):
            noisy_query = pattern.sub(replacement, noisy_query, count=1)
            consumed_anchor = True
            break
    if not consumed_anchor:
        # Just replace the identifier directly
        noisy_query = _case_insensitive_replace(noisy_query, str(pval), replacement)
    return NoisyInstance(
        instance_id=f"{s.source_query_id}__AMBIGUOUS_REFERENCE__{param}",
        source_query_id=s.source_query_id,
        noise_category="AMBIGUOUS_REFERENCE",
        noisy_query=noisy_query,
        original_query=s.query,
        tool_registry=s.tool_registry,
        gold_tool_name=s.gold_tool_name,
        gold_arguments=s.gold_arguments,
        parameter_schema=s.parameter_schema,
        noisy_parameter=param,
        noisy_parameter_gold_value=pval,
        expected_clarification_target=param,
    )


def _wrong_premise(s: CleanSource, param: str, rng: random.Random) -> NoisyInstance:
    pval = s.gold_arguments[param]
    schema = s.parameter_schema.get(param, {})
    enum = schema.get("enum", [])
    alternatives = [v for v in enum if v != pval]
    if not alternatives:
        raise ValueError(f"No alternative enum values for {param}")
    wrong_val = rng.choice(alternatives)
    noisy_query = _case_insensitive_replace(s.query, str(pval), str(wrong_val))
    return NoisyInstance(
        instance_id=f"{s.source_query_id}__WRONG_PREMISE__{param}",
        source_query_id=s.source_query_id,
        noise_category="WRONG_PREMISE",
        noisy_query=noisy_query,
        original_query=s.query,
        tool_registry=s.tool_registry,
        gold_tool_name=s.gold_tool_name,
        gold_arguments=s.gold_arguments,
        parameter_schema=s.parameter_schema,
        noisy_parameter=param,
        noisy_parameter_gold_value=pval,  # the CORRECT value, not the wrong one
        expected_clarification_target=param,
    )


def _underspecified_type(s: CleanSource, param: str, rng: random.Random) -> NoisyInstance:
    # param is the sentinel "__TOOL__" — we noise the tool choice via the query
    gold_name = s.gold_tool_name.lower()
    prefix_map = {
        "send_": "send",
        "get_": "fetch",
        "fetch_": "fetch",
        "post_": "post",
        "save_": "save",
        "export_": "export",
    }
    # Handle Tool__API naming
    api_part = gold_name.split("__", 1)[1] if "__" in gold_name else gold_name
    # Find which prefix matches
    matched_prefix = None
    for p in prefix_map:
        if api_part.startswith(p):
            matched_prefix = p
            break
    if not matched_prefix:
        raise ValueError(f"Cannot find prefix for {gold_name}")
    # Strategy: find a "by <medium>" or "via <medium>" or "as <format>"
    # phrase and remove it entirely. If no such phrase, fall back to
    # replacing the standalone medium word with a generic verb.
    noisy_query = s.query
    replaced = False
    # First try to remove "by/via/as <medium>" phrases
    medium_phrase_pattern = re.compile(
        r'\s*\b(by|via|as|using|through)\s+(an?\s+)?'
        r'(email|sms|text|slack|message|pdf|csv|json|xml|spreadsheet)\b',
        re.IGNORECASE,
    )
    if medium_phrase_pattern.search(noisy_query):
        noisy_query = medium_phrase_pattern.sub("", noisy_query, count=1)
        noisy_query = re.sub(r'\s+', ' ', noisy_query).strip()
        noisy_query = re.sub(r'\s+([.,!?])', r'\1', noisy_query)
        replaced = True
    if not replaced:
        # Fall back to replacing a standalone medium word
        specifier_words = {
            "email": "message", "sms": "message", "slack": "message",
            "pdf": "file", "csv": "file", "json": "file",
        }
        for spec, neutral in specifier_words.items():
            pattern = re.compile(r'\b' + re.escape(spec) + r'\b', re.IGNORECASE)
            if pattern.search(noisy_query):
                noisy_query = pattern.sub(neutral, noisy_query, count=1)
                replaced = True
                break
    # If still not replaced, the query is left as-is (we keep this instance
    # but note it in the metadata so the analysis can flag it)
    return NoisyInstance(
        instance_id=f"{s.source_query_id}__UNDERSPECIFIED_TYPE__tool",
        source_query_id=s.source_query_id,
        noise_category="UNDERSPECIFIED_TYPE",
        noisy_query=noisy_query,
        original_query=s.query,
        tool_registry=s.tool_registry,
        gold_tool_name=s.gold_tool_name,
        gold_arguments=s.gold_arguments,
        parameter_schema=s.parameter_schema,
        noisy_parameter="__TOOL__",
        noisy_parameter_gold_value=s.gold_tool_name,
        expected_clarification_target="tool_choice",
    )


def _volume_scope(s: CleanSource, param: str, rng: random.Random) -> NoisyInstance:
    pval = s.gold_arguments[param]
    pname_lower = param.lower()
    # Pick a noun root that matches the parameter
    noun_root = None
    for root in VOLUME_PLURALISATIONS:
        if root in pname_lower:
            noun_root = root
            break
    if noun_root is None:
        noun_root = "item"
    plural = VOLUME_PLURALISATIONS[noun_root]
    noisy_query = s.query

    # Strategy 1: match "(the|my|a|an)? <noun> #?<value>" and replace with
    # "my <plural>". Absorbing the determiner ensures we don't leave "the"
    # stranded before "my companies".
    handled = False
    pattern_with_det = re.compile(
        r'\b(?:the|my|a|an)\s+' + noun_root + r'\s+#?' + re.escape(str(pval)) + r'\b',
        re.IGNORECASE,
    )
    if pattern_with_det.search(noisy_query):
        noisy_query = pattern_with_det.sub(f'my {plural}', noisy_query, count=1)
        handled = True

    if not handled:
        # Strategy 1b: no determiner, just "<noun> <value>"
        pattern_bare = re.compile(
            r'\b' + noun_root + r'\s+#?' + re.escape(str(pval)) + r'\b',
            re.IGNORECASE,
        )
        if pattern_bare.search(noisy_query):
            noisy_query = pattern_bare.sub(f'my {plural}', noisy_query, count=1)
            handled = True

    if not handled:
        # Strategy 2: remove the value and pluralise the noun
        noisy_query = _case_insensitive_replace(noisy_query, str(pval), "")
        # Replace "the/my/a/an <noun>" with "my <plural>"
        det_pattern = re.compile(
            r'\b(the|my|a|an)\s+' + noun_root + r'\b',
            re.IGNORECASE,
        )
        if det_pattern.search(noisy_query):
            noisy_query = det_pattern.sub(f'my {plural}', noisy_query, count=1)
        else:
            # Strategy 3: just pluralise the noun in place
            noisy_query = re.sub(
                r'\b' + noun_root + r'\b',
                plural,
                noisy_query, count=1, flags=re.IGNORECASE,
            )

    # Clean up whitespace and stray punctuation
    noisy_query = re.sub(r'\s+', ' ', noisy_query).strip()
    noisy_query = re.sub(r'\s+([.,!?])', r'\1', noisy_query)
    return NoisyInstance(
        instance_id=f"{s.source_query_id}__VOLUME_SCOPE__{param}",
        source_query_id=s.source_query_id,
        noise_category="VOLUME_SCOPE",
        noisy_query=noisy_query,
        original_query=s.query,
        tool_registry=s.tool_registry,
        gold_tool_name=s.gold_tool_name,
        gold_arguments=s.gold_arguments,
        parameter_schema=s.parameter_schema,
        noisy_parameter=param,
        noisy_parameter_gold_value=pval,
        expected_clarification_target=param,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
TRANSFORMS = {
    "MISSING_REQUIRED": _missing_required,
    "AMBIGUOUS_REFERENCE": _ambiguous_reference,
    "WRONG_PREMISE": _wrong_premise,
    # UNDERSPECIFIED_TYPE and VOLUME_SCOPE removed from v1.1 — see ELIGIBILITY
    # comment above. Functions `_underspecified_type` and `_volume_scope`
    # remain in the file for v1.2 / alternative source corpora.
}


def apply_noise(s: CleanSource, category: str, seed: int) -> NoisyInstance:
    """Apply a noise transform deterministically."""
    eligible_params = ELIGIBILITY[category](s)
    if not eligible_params:
        raise ValueError(
            f"Source {s.source_query_id} not eligible for {category}"
        )
    rng = random.Random((s.source_query_id, category, seed).__hash__())
    param = rng.choice(eligible_params)
    return TRANSFORMS[category](s, param, rng)
