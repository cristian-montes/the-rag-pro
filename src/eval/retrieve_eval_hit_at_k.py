import json
import os
from rapidfuzz import fuzz
from retrieval import retrieve

# Evaluation config
EVAL_FILE = "src/eval/retrieve_eval_set.json"
OUTPUT_DIR = "eval"
TOP_KS = [1, 3, 5]
FUZZY_THRESHOLD = 0.7  # Typically 0.7 to 0.85 is a good range


os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fuzzy matching logic using RapidFuzz token_sort_ratio
def fuzzy_match(text, expected, threshold=FUZZY_THRESHOLD):
    # Compute similarity score (0-100), convert to 0-1 range
    score = fuzz.partial_ratio(text.lower(), expected.lower()) / 100
    return score >= threshold, score

# Match helper with logging
def matches(text, expected_answers, threshold=FUZZY_THRESHOLD):
    for expected in expected_answers:
        matched, score = fuzzy_match(text, expected, threshold)
        if matched:
            return True, expected, score
    return False, None, 0.0

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
        "hit@": {},
        "details": []
    }

    for k in TOP_KS:
        top_hits = hits[:k]
        matched = False

        for h in top_hits:
            is_hit, matched_str, score = matches(h["doc"], expected)
            result["details"].append({
                "k": k,
                "chunk": h["doc"],
                "matched": is_hit,
                "matched_with": matched_str,
                "similarity": round(score, 3)
            })
            if is_hit:
                matched = True
                break  # Stop checking once we find a hit for this K

        result["hit@"][f"@{k}"] = matched
        if matched:
            hit_counts[k] += 1

    results.append(result)

# Print summary
print("📊 Results:")
for k in TOP_KS:
    score = hit_counts[k] / len(eval_set)
    print(f"Hit@{k}: {hit_counts[k]} / {len(eval_set)} = {score:.2%}")

# Save detailed results to disk
with open(os.path.join(OUTPUT_DIR, "hit_at_k_results.json"), "w") as f:
    json.dump(results, f, indent=2)
