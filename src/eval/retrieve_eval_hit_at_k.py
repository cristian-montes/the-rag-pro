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















# import json
# import os
# from rapidfuzz import fuzz
# from retrieval import retrieve

# # Config
# EVAL_FILE = "src/eval/retrieve_eval_set.json"
# OUTPUT_DIR = "eval"
# TOP_KS = [1, 3, 5]
# FUZZY_THRESHOLD = 0.7

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # --- Helpers ---

# def fuzzy_match(text, expected, threshold=FUZZY_THRESHOLD):
#     score = fuzz.partial_ratio(text.lower(), expected.lower()) / 100
#     return score >= threshold, score

# def matches(text, expected_answers, threshold=FUZZY_THRESHOLD):
#     for expected in expected_answers:
#         matched, score = fuzzy_match(text, expected, threshold)
#         if matched:
#             return True, expected, score
#     return False, None, 0.0

# def split_into_claims(answer_text):
#     # Split on '.', ';' or newline, and clean
#     import re
#     raw_claims = re.split(r'[.;\n]', answer_text)
#     return [claim.strip() for claim in raw_claims if len(claim.strip()) > 10]  # ignore short fragments

# # --- Load eval set ---

# with open(EVAL_FILE, "r") as f:
#     eval_set = json.load(f)

# # Normalize
# for item in eval_set:
#     if isinstance(item["expected"], str):
#         item["expected"] = [item["expected"]]

# hit_counts = {k: 0 for k in TOP_KS}
# context_precisions = []
# context_recalls = []
# results = []

# print("🔍 Running Hit@K + Context Recall/Precision...\n")

# for item in eval_set:
#     query = item["query"]
#     expected_list = item["expected"]
#     expected_text = " ".join(expected_list)
#     claims = split_into_claims(expected_text)

#     hits = retrieve(query, k=max(TOP_KS))
#     retrieved_docs = [h["doc"] for h in hits]

#     # --- Context Recall ---
#     matched_claims = 0
#     for claim in claims:
#         for chunk in retrieved_docs:
#             matched, _ = fuzzy_match(chunk, claim)
#             if matched:
#                 matched_claims += 1
#                 break  # claim matched in at least one doc
#     recall = matched_claims / len(claims) if claims else 0
#     context_recalls.append(recall)

#     # --- Context Precision ---
#     supporting_chunks = 0
#     for chunk in retrieved_docs:
#         for claim in claims:
#             matched, _ = fuzzy_match(chunk, claim)
#             if matched:
#                 supporting_chunks += 1
#                 break  # this chunk supports at least one claim
#     precision = supporting_chunks / len(retrieved_docs) if retrieved_docs else 0
#     context_precisions.append(precision)

#     # --- Hit@K ---
#     result = {
#         "query": query,
#         "expected": expected_list,
#         "retrieved": retrieved_docs,
#         "hit@": {},
#         "context_recall": round(recall, 3),
#         "context_precision": round(precision, 3),
#         "details": []
#     }

#     for k in TOP_KS:
#         top_hits = hits[:k]
#         matched = False

#         for h in top_hits:
#             is_hit, matched_str, score = matches(h["doc"], expected_list)
#             result["details"].append({
#                 "k": k,
#                 "chunk": h["doc"],
#                 "matched": is_hit,
#                 "matched_with": matched_str,
#                 "similarity": round(score, 3)
#             })
#             if is_hit:
#                 matched = True
#                 break

#         result["hit@"][f"@{k}"] = matched
#         if matched:
#             hit_counts[k] += 1

#     results.append(result)

# # --- Final Output ---

# print("📊 Results:")
# for k in TOP_KS:
#     score = hit_counts[k] / len(eval_set)
#     print(f"Hit@{k}: {hit_counts[k]} / {len(eval_set)} = {score:.2%}")

# avg_recall = sum(context_recalls) / len(context_recalls)
# avg_precision = sum(context_precisions) / len(context_precisions)

# print(f"\n📘 Context Recall:    {avg_recall:.2%}")
# print(f"📗 Context Precision: {avg_precision:.2%}")

# # Save detailed results
# with open(os.path.join(OUTPUT_DIR, "retriever_eval_results.json"), "w") as f:
#     json.dump(results, f, indent=2)







# # import json
# # import os
# # from rapidfuzz import fuzz
# # from retrieval import retrieve

# # # Evaluation config
# # EVAL_FILE = "src/eval/retrieve_eval_set.json"
# # OUTPUT_DIR = "eval"
# # TOP_KS = [1, 3, 5]
# # FUZZY_THRESHOLD = 0.7  # Typically 0.7 to 0.85 is a good range for fuzzy matching


# # os.makedirs(OUTPUT_DIR, exist_ok=True)

# # # Fuzzy matching logic using RapidFuzz token_sort_ratio
# # def fuzzy_match(text, expected, threshold=FUZZY_THRESHOLD):
# #     # Compute similarity score (0-100), convert to 0-1 range
# #     score = fuzz.partial_ratio(text.lower(), expected.lower()) / 100
# #     return score >= threshold, score

# # # Match helper with logging
# # def matches(text, expected_answers, threshold=FUZZY_THRESHOLD):
# #     for expected in expected_answers:
# #         matched, score = fuzzy_match(text, expected, threshold)
# #         if matched:
# #             return True, expected, score
# #     return False, None, 0.0

# # # Load eval set
# # with open(EVAL_FILE, "r") as f:
# #     eval_set = json.load(f)

# # # Normalize expected to always be a list
# # for item in eval_set:
# #     if isinstance(item["expected"], str):
# #         item["expected"] = [item["expected"]]

# # # Initialize hit counters
# # hit_counts = {k: 0 for k in TOP_KS}
# # results = []

# # print("🔍 Running Hit@K evaluation...\n")

# # # Run evaluation
# # for item in eval_set:
# #     query = item["query"]
# #     expected = item["expected"]
# #     hits = retrieve(query, k=max(TOP_KS))

# #     result = {
# #         "query": query,
# #         "expected": expected,
# #         "retrieved": [h["doc"] for h in hits],
# #         "hit@": {},
# #         "details": []
# #     }

# #     for k in TOP_KS:
# #         top_hits = hits[:k]
# #         matched = False

# #         for h in top_hits:
# #             is_hit, matched_str, score = matches(h["doc"], expected)
# #             result["details"].append({
# #                 "k": k,
# #                 "chunk": h["doc"],
# #                 "matched": is_hit,
# #                 "matched_with": matched_str,
# #                 "similarity": round(score, 3)
# #             })
# #             if is_hit:
# #                 matched = True
# #                 break  # Stop checking once we find a hit for this K

# #         result["hit@"][f"@{k}"] = matched
# #         if matched:
# #             hit_counts[k] += 1

# #     results.append(result)

# # # Print summary
# # print("📊 Results:")
# # for k in TOP_KS:
# #     score = hit_counts[k] / len(eval_set)
# #     print(f"Hit@{k}: {hit_counts[k]} / {len(eval_set)} = {score:.2%}")

# # # Save detailed results to disk
# # with open(os.path.join(OUTPUT_DIR, "hit_at_k_results.json"), "w") as f:
# #     json.dump(results, f, indent=2)
