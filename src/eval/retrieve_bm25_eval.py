import json
import os
import re
from rapidfuzz import fuzz
import spacy
from sparse.retrieval_bm25 import retrieve  # or your BM25 retriever method

# --- Config ---
EVAL_FILE = "src/eval/retrieve_bm25_eval_set.json"
OUTPUT_DIR = "eval"
TOP_K = 5
FUZZY_THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load spaCy for lemmatization ---
nlp = spacy.load("en_core_web_sm")

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc if token.is_alpha)

# --- Helpers ---
def fuzzy_or_containment_match(chunk, expected, threshold=FUZZY_THRESHOLD):
    chunk_lower = chunk.lower()
    expected_lower = expected.lower()

    # Check if expected string is a substring of the chunk
    if expected_lower in chunk_lower:
        return True, 1.0

    # Fallback to fuzzy match
    score = fuzz.partial_ratio(chunk_lower, expected_lower) / 100
    return score >= threshold, score

def split_into_claims(answer_text):
    raw_claims = re.split(r'[.;\n]', answer_text)
    return [claim.strip() for claim in raw_claims if len(claim.strip()) > 10]

# --- Load and Normalize Eval Set ---
with open(EVAL_FILE, "r") as f:
    eval_set = json.load(f)

for item in eval_set:
    if isinstance(item.get("expected"), str):
        item["expected"] = [item["expected"]]

# --- Run Evaluation ---
context_precisions = []
context_recalls = []
results = []

print("🔍 Running Context Precision & Recall Evaluation...\n")

for item in eval_set:
    raw_query = item["query"]
    expected_list = item["expected"]

    # ✅ Fix 1: Lemmatize query to match lemmatized chunks
    lemmatized_query = lemmatize_text(raw_query)

    hits = retrieve(lemmatized_query, k=TOP_K)
    retrieved_docs = [h["doc"] for h in hits]

    expected_text = " ".join(expected_list)
    claims = split_into_claims(expected_text)

    # --- Context Recall ---
    matched_claims = 0
    for claim in claims:
        for chunk in retrieved_docs:
            matched, _ = fuzzy_or_containment_match(chunk, claim)
            if matched:
                matched_claims += 1
                break
    recall = matched_claims / len(claims) if claims else 0
    context_recalls.append(recall)

    # --- Context Precision ---
    supporting_chunks = 0
    for chunk in retrieved_docs:
        for claim in claims:
            matched, _ = fuzzy_or_containment_match(chunk, claim)
            if matched:
                supporting_chunks += 1
                break
    precision = supporting_chunks / len(retrieved_docs) if retrieved_docs else 0
    context_precisions.append(precision)

    results.append({
        "query": raw_query,
        "expected": expected_list,
        "retrieved": retrieved_docs,
        "context_recall": round(recall, 3),
        "context_precision": round(precision, 3),
        "claims": claims
    })

# --- Summary ---
avg_recall = sum(context_recalls) / len(context_recalls)
avg_precision = sum(context_precisions) / len(context_precisions)

print(f"\n📘 Context Recall (avg):    {avg_recall:.2%}")
print(f"📗 Context Precision (avg): {avg_precision:.2%}")

# --- Save Output ---
with open(os.path.join(OUTPUT_DIR, "context_eval_results.json"), "w") as f:
    json.dump(results, f, indent=2)




