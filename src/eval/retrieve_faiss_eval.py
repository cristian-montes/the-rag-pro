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
print("🔍 Running Simple Semantic Similarity Evaluation...\n")

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

print(f"\n✅ Evaluation complete. Results saved to: {output_path}")













# import json
# import os
# import re
# from sentence_transformers import SentenceTransformer, util
# from retrieval import retrieve
# import argparse

# # -----------------------------
# # Argument Parser
# # -----------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument("--eval-file", default="src/eval/retrieve_faiss_eval_set.json")
# parser.add_argument("--output-dir", default="eval/results")
# parser.add_argument("--top-k", type=int, default=5)
# parser.add_argument("--threshold", type=float, default=0.7)
# args = parser.parse_args()

# # -----------------------------
# # Constants
# # -----------------------------
# EVAL_FILE = args.eval_file
# OUTPUT_DIR = args.output_dir
# TOP_K = args.top_k
# THRESHOLD = args.threshold

# # -----------------------------
# # Load Evaluation Set
# # -----------------------------
# with open(EVAL_FILE) as f:
#     eval_set = json.load(f)

# # Normalize expected to list
# for item in eval_set:
#     if isinstance(item["expected"], str):
#         item["expected"] = [item["expected"]]

# # -----------------------------
# # Setup
# # -----------------------------
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# context_precisions = []
# context_recalls = []
# results = []

# # -----------------------------
# # Helper
# # -----------------------------
# def split_into_claims(text):
#     raw_claims = re.split(r'[.;\n]', text)
#     return [c.strip() for c in raw_claims if len(c.strip()) > 10]

# def cosine_match(text1, text2, threshold=THRESHOLD):
#     embs = model.encode([text1, text2], convert_to_tensor=True)
#     score = util.cos_sim(embs[0], embs[1]).item()
#     return score >= threshold, score

# # -----------------------------
# # Evaluation Loop
# # -----------------------------
# print("🔍 Running FAISS Cosine Evaluation...\n")

# for item in eval_set:
#     query = item["query"]
#     expected_list = item["expected"]
#     expected_text = " ".join(expected_list)
#     claims = split_into_claims(expected_text)

#     hits = retrieve(query, k=TOP_K)
#     retrieved_docs = [hit["doc"] for hit in hits]

#     # --- Context Recall ---
#     matched_claims = 0
#     for claim in claims:
#         for chunk in retrieved_docs:
#             matched, _ = cosine_match(chunk, claim)
#             if matched:
#                 matched_claims += 1
#                 break
#     recall = matched_claims / len(claims) if claims else 0
#     context_recalls.append(recall)

#     # --- Context Precision ---
#     supporting_chunks = 0
#     for chunk in retrieved_docs:
#         for claim in claims:
#             matched, _ = cosine_match(chunk, claim)
#             if matched:
#                 supporting_chunks += 1
#                 break
#     precision = supporting_chunks / len(retrieved_docs) if retrieved_docs else 0
#     context_precisions.append(precision)

#     results.append({
#         "query": query,
#         "expected": expected_list,
#         "retrieved": retrieved_docs,
#         "context_recall": round(recall, 3),
#         "context_precision": round(precision, 3),
#         "claims": claims
#     })

# # -----------------------------
# # Final Metrics
# # -----------------------------
# avg_recall = sum(context_recalls) / len(context_recalls)
# avg_precision = sum(context_precisions) / len(context_precisions)

# print("\n--- Evaluation Results ---")
# print(f"📘 Context Recall (avg):    {avg_recall:.2%}")
# print(f"📗 Context Precision (avg): {avg_precision:.2%}")

# # -----------------------------
# # Save Output
# # -----------------------------
# with open(os.path.join(OUTPUT_DIR, "context_eval_faiss_results.json"), "w") as f:
#     json.dump(results, f, indent=2)
