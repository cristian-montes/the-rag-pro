import json
import os
import re
from rapidfuzz import fuzz
from retrieval import retrieve

# Config
EVAL_FILE = "src/eval/retrieve_eval_set.json"
OUTPUT_DIR = "eval"
TOP_K = 5
FUZZY_THRESHOLD = 0.7

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helpers ---

def fuzzy_match(text, expected, threshold=FUZZY_THRESHOLD):
    score = fuzz.partial_ratio(text.lower(), expected.lower()) / 100
    return score >= threshold, score

def split_into_claims(answer_text):
    # Split on '.', ';' or newline, and clean
    raw_claims = re.split(r'[.;\n]', answer_text)
    return [claim.strip() for claim in raw_claims if len(claim.strip()) > 10]  # ignore short fragments

# --- Load eval set ---

with open(EVAL_FILE, "r") as f:
    eval_set = json.load(f)

# Normalize
for item in eval_set:
    if isinstance(item["expected"], str):
        item["expected"] = [item["expected"]]

context_precisions = []
context_recalls = []
results = []

print("🔍 Running Context Precision & Recall Evaluation...\n")

# --- Evaluation loop ---

for item in eval_set:
    query = item["query"]
    expected_list = item["expected"]
    expected_text = " ".join(expected_list)
    claims = split_into_claims(expected_text)

    hits = retrieve(query, k=TOP_K)
    retrieved_docs = [h["doc"] for h in hits]

    # --- Context Recall ---
    matched_claims = 0
    for claim in claims:
        for chunk in retrieved_docs:
            matched, _ = fuzzy_match(chunk, claim)
            if matched:
                matched_claims += 1
                break
    recall = matched_claims / len(claims) if claims else 0
    context_recalls.append(recall)

    # --- Context Precision ---
    supporting_chunks = 0
    for chunk in retrieved_docs:
        for claim in claims:
            matched, _ = fuzzy_match(chunk, claim)
            if matched:
                supporting_chunks += 1
                break
    precision = supporting_chunks / len(retrieved_docs) if retrieved_docs else 0
    context_precisions.append(precision)

    results.append({
        "query": query,
        "expected": expected_list,
        "retrieved": retrieved_docs,
        "context_recall": round(recall, 3),
        "context_precision": round(precision, 3),
        "claims": claims
    })

# --- Print final results ---

avg_recall = sum(context_recalls) / len(context_recalls)
avg_precision = sum(context_precisions) / len(context_precisions)

print(f"📘 Context Recall (avg):    {avg_recall:.2%}")
print(f"📗 Context Precision (avg): {avg_precision:.2%}")

# --- Save output ---
with open(os.path.join(OUTPUT_DIR, "context_eval_results.json"), "w") as f:
    json.dump(results, f, indent=2)

