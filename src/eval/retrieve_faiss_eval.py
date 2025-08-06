import json
import os
from sentence_transformers import SentenceTransformer, util
from src.dense.retrieval import retrieve
import argparse

# -----------------------------
# Argument Parser
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--eval-file", default="src/eval/retrieve_faiss_eval_set.json")
parser.add_argument("--output-dir", default="eval/results")
parser.add_argument("--top-k", type=int, default=5)
args = parser.parse_args()

# -----------------------------
# Constants
# -----------------------------
EVAL_FILE = args.eval_file
OUTPUT_DIR = args.output_dir
TOP_K = args.top_k

# -----------------------------
# Setup
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# Load Evaluation Set
# -----------------------------
with open(EVAL_FILE) as f:
    eval_set = json.load(f)

# Normalize expected to list
for item in eval_set:
    if isinstance(item["expected"], str):
        item["expected"] = [item["expected"]]

results = []

# -----------------------------
# Evaluation Loop
# -----------------------------
print(" Running Simple Semantic Similarity Evaluation...\n")

for item in eval_set:
    query = item["query"]
    expected_list = item["expected"]

    hits = retrieve(query, k=TOP_K)
    retrieved_chunks = [hit["doc"] for hit in hits]

    # If no retrieved chunks, similarity is zero
    if not retrieved_chunks:
        avg_similarity = 0.0
        per_expected_similarities = [0.0 for _ in expected_list]
    else:
        per_expected_similarities = []
        # For each expected answer
        for expected_text in expected_list:
            similarities = []
            # Compare each retrieved chunk to this expected answer
            for chunk in retrieved_chunks:
                emb1, emb2 = model.encode([expected_text, chunk], convert_to_tensor=True)
                score = util.cos_sim(emb1, emb2).item()
                similarities.append(score)
            # Average similarity for this expected answer over all retrieved chunks
            per_expected_similarities.append(sum(similarities) / len(similarities))
        # Average over all expected answers
        avg_similarity = sum(per_expected_similarities) / len(per_expected_similarities)

    results.append({
        "query": query,
        "expected": expected_list,
        "retrieved": retrieved_chunks,
        "per_expected_similarity": [round(s, 4) for s in per_expected_similarities],
        "avg_similarity": round(avg_similarity, 4)
    })

# -----------------------------
# Save Output
# -----------------------------
output_path = os.path.join(OUTPUT_DIR, "semantic_eval_results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n Evaluation complete. Results saved to: {output_path}")