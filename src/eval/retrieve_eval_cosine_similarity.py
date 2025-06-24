import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Load eval dataset ----
# Format: list of {"query": ..., "expected": ...}
with open("eval/eval_set.json", "r") as f:
    eval_set = json.load(f)

# ---- Load corpus chunks ----
with open("index/bm25_corpus.json", "r") as f:
    corpus_chunks = json.load(f)

# ---- TF-IDF Vectorizer ----
vectorizer = TfidfVectorizer().fit(corpus_chunks)

# Vectorize corpus
corpus_vecs = vectorizer.transform(corpus_chunks)

# ---- Evaluate similarity ----
print("🔍 Running cosine similarity eval...\n")

results = []
for item in eval_set:
    query = item["query"]
    expected = item["expected"]

    # Vectorize expected answer
    expected_vec = vectorizer.transform([expected])

    # Compute similarity to all chunks
    sims = cosine_similarity(expected_vec, corpus_vecs)[0]

    # Sort top-k scores
    top_k_indices = sims.argsort()[::-1][:5]
    top_k_scores = [round(sims[i], 3) for i in top_k_indices]
    top_k_chunks = [corpus_chunks[i] for i in top_k_indices]

    results.append({
        "query": query,
        "expected": expected,
        "top_k_scores": top_k_scores,
        "top_k_chunks": top_k_chunks
    })

# ---- Output results ----
with open("eval/cosine_eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ Cosine similarity eval complete. Saved to eval/cosine_eval_results.json")
