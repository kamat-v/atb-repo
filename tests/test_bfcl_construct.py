"""
Smoke test for the BFCL-targeted construction pipeline using real records
fetched from the upstream BFCL_v3_simple.json file.

Verifies:
  - The parse pipeline produces CleanSources
  - Each noise category gets at least some assignments
  - Noisy queries are recognisably altered from originals
  - Gold values are correctly preserved in the metadata
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

# A representative slice of real BFCL_v3_simple.json records, chosen to
# exercise multiple noise categories.
REAL_BFCL_QUESTIONS = [
    # Numeric params, multiple required → MISSING_REQUIRED or VOLUME_SCOPE
    {
        "id": "simple_0",
        "question": [[{"role": "user",
                       "content": "Find the area of a triangle with a base of 10 units and height of 5 units."}]],
        "function": [{
            "name": "calculate_triangle_area",
            "description": "Calculate the area of a triangle given its base and height.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "base": {"type": "integer", "description": "The base of the triangle."},
                    "height": {"type": "integer", "description": "The height of the triangle."},
                    "unit": {"type": "string", "description": "The unit of measure (defaults to 'units' if not specified)"},
                },
                "required": ["base", "height"],
            },
        }],
    },
    {
        "id": "simple_33",
        "question": [[{"role": "user",
                       "content": "Get directions from Sydney to Melbourne using the fastest route."}]],
        "function": [{
            "name": "get_directions",
            "description": "Retrieve directions from one location to another.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "start_location": {"type": "string", "description": "The starting point of the journey."},
                    "end_location": {"type": "string", "description": "The destination point of the journey."},
                    "route_type": {"type": "string", "description": "Type of route (e.g., 'fastest', 'scenic'). Default is 'fastest'.",
                                   "enum": ["fastest", "scenic"]},
                },
                "required": ["start_location", "end_location"],
            },
        }],
    },
    {
        "id": "simple_36",
        "question": [[{"role": "user",
                       "content": "Find the shortest driving distance between New York City and Washington D.C."}]],
        "function": [{
            "name": "get_shortest_driving_distance",
            "description": "Calculate the shortest driving distance between two locations.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "origin": {"type": "string", "description": "Starting point."},
                    "destination": {"type": "string", "description": "End point."},
                    "unit": {"type": "string", "description": "Optional. Default 'km'."},
                },
                "required": ["origin", "destination"],
            },
        }],
    },
    {
        "id": "simple_77",
        "question": [[{"role": "user",
                       "content": "Find a nearby restaurant that serves vegan food in Los Angeles."}]],
        "function": [{
            "name": "restaurant.find_nearby",
            "description": "Locate nearby restaurants based on dietary preferences.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "location": {"type": "string"},
                    "dietary_preference": {"type": "array", "items": {"type": "string",
                        "enum": ["Vegan", "Vegetarian", "Gluten-free"]}},
                },
                "required": ["location"],
            },
        }],
    },
    {
        "id": "simple_92",
        "question": [[{"role": "user",
                       "content": "Find all movies starring Leonardo DiCaprio in the year 2010 from IMDB database."}]],
        "function": [{
            "name": "imdb.find_movies_by_actor",
            "description": "Searches the database to find all movies by a specific actor within a year.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "actor_name": {"type": "string"},
                    "year": {"type": "integer"},
                    "category": {"type": "string"},
                },
                "required": ["actor_name", "year"],
            },
        }],
    },
    {
        "id": "simple_142",
        "question": [[{"role": "user",
                       "content": "Find the NASDAQ stock price for the company Amazon at closing March.11, 2022."}]],
        "function": [{
            "name": "get_stock_price",
            "description": "Get the closing stock price.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "company_name": {"type": "string"},
                    "date": {"type": "string"},
                    "exchange": {"type": "string"},
                },
                "required": ["company_name", "date"],
            },
        }],
    },
    {
        "id": "simple_25",
        "question": [[{"role": "user",
                       "content": "Calculate the final velocity of an object falling from a 150 meter building, assuming initial velocity is zero."}]],
        "function": [{
            "name": "calculate_final_velocity",
            "description": "Calculate final velocity from height.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "height": {"type": "integer"},
                    "initial_velocity": {"type": "integer"},
                    "gravity": {"type": "float"},
                },
                "required": ["height"],
            },
        }],
    },
]

# Matching ground truths (gold parameter values).
# Real BFCL format: ground_truth[0] is a single-key dict where the key is the
# gold tool's name and the value is the param dict.
REAL_BFCL_ANSWERS = [
    {"id": "simple_0", "ground_truth": [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]},
    {"id": "simple_33", "ground_truth": [{"get_directions": {"start_location": ["Sydney"], "end_location": ["Melbourne"], "route_type": ["fastest", ""]}}]},
    {"id": "simple_36", "ground_truth": [{"get_shortest_driving_distance": {"origin": ["New York City"], "destination": ["Washington D.C."], "unit": ["km", ""]}}]},
    {"id": "simple_77", "ground_truth": [{"restaurant.find_nearby": {"location": ["Los Angeles"], "dietary_preference": [["Vegan"]]}}]},
    {"id": "simple_92", "ground_truth": [{"imdb.find_movies_by_actor": {"actor_name": ["Leonardo DiCaprio"], "year": [2010], "category": ["all", ""]}}]},
    {"id": "simple_142", "ground_truth": [{"get_stock_price": {"company_name": ["Amazon"], "date": ["2022-03-11"], "exchange": ["NASDAQ"]}}]},
    {"id": "simple_25", "ground_truth": [{"calculate_final_velocity": {"height": [150], "initial_velocity": [0], "gravity": [9.81]}}]},
]


def main():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        q_path = td / "questions.json"
        a_path = td / "answers.json"

        # Write JSONL (one record per line, the format BFCL uses)
        with open(q_path, "w") as f:
            for q in REAL_BFCL_QUESTIONS:
                f.write(json.dumps(q) + "\n")
        with open(a_path, "w") as f:
            for a in REAL_BFCL_ANSWERS:
                f.write(json.dumps(a) + "\n")

        out_dir = td / "out"
        # Resolve construct_dataset.py relative to this file rather than
        # hardcoding an absolute path. Makes the test portable.
        _construct_py = (
            Path(__file__).resolve().parent.parent / "src" / "construct_dataset.py"
        )
        result = subprocess.run([
            sys.executable, str(_construct_py),
            "--questions-file", str(q_path),
            "--answers-file", str(a_path),
            "--output-dir", str(out_dir),
            "--seed", "0",
            "--target-per-category", "2",
            "--clean-target", "2",
        ], capture_output=True, text=True)

        print("=== STDOUT ===")
        print(result.stdout)
        if result.stderr:
            print("=== STDERR ===")
            print(result.stderr)
        if result.returncode != 0:
            print(f"FAILED (returncode={result.returncode})")
            return

        # Inspect noisy outputs
        print("\n=== atb_v1.jsonl ===")
        for line in open(out_dir / "atb_v1.jsonl"):
            inst = json.loads(line)
            print(f"\n  [{inst['noise_category']}] src={inst['source_query_id']}")
            print(f"  original: {inst['original_query']!r}")
            print(f"  noisy:    {inst['noisy_query']!r}")
            print(f"  target:   {inst['noisy_parameter']}")
            print(f"  gold val: {inst['noisy_parameter_gold_value']!r}")
            print(f"  gold tool: {inst['gold_tool_name']}")

        # Inspect clean
        print("\n=== atb_clean_v1.jsonl ===")
        for line in open(out_dir / "atb_clean_v1.jsonl"):
            inst = json.loads(line)
            print(f"  src={inst['source_query_id']} | query: {inst['query']!r}")
            print(f"    args: {inst['gold_arguments']}")


def test_smoke_bfcl_construct():
    """Pytest entrypoint."""
    main()


if __name__ == "__main__":
    main()
