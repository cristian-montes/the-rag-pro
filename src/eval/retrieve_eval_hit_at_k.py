import json
import os
from retrieval import retrieve

# Evaluation config
EVAL_FILE = "src/eval/retrieve_eval_set.json"
OUTPUT_DIR = "eval"
TOP_KS = [1, 3, 5]

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper: check if any expected string is found in the text
def matches(text, expected_answers):
    text = text.lower()
    return any(expected.lower() in text for expected in expected_answers)

# Load eval set
with open(EVAL_FILE, "r") as f:
    eval_set = json.load(f)

# Normalize expected to always be a list
for item in eval_set:
    if isinstance(item["expected"], str):
        item["expected"] = [item["expected"]]

# Initialize hit counters
hit_counts = {k: 0 for k in TOP_KS}
results = []

print("🔍 Running Hit@K evaluation...\n")

# Run evaluation
for item in eval_set:
    query = item["query"]
    expected = item["expected"]
    hits = retrieve(query, k=max(TOP_KS))

    result = {
        "query": query,
        "expected": expected,
        "retrieved": [h["doc"] for h in hits],
        "hit@": {}
    }

    for k in TOP_KS:
        top_hits = hits[:k]
        matched = any(matches(h["doc"], expected) for h in top_hits)
        result["hit@"][f"@{k}"] = matched
        if matched:
            hit_counts[k] += 1

    results.append(result)

# Print summary
print("📊 Results:")
for k in TOP_KS:
    score = hit_counts[k] / len(eval_set)
    print(f"Hit@{k}: {hit_counts[k]} / {len(eval_set)} = {score:.2%}")

# Save
