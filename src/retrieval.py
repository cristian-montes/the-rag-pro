import json
import os
import pickle
import numpy as np
from rank_bm25 import BM25Okapi

# Path to saved index directory
INDEX_DIR = "index"

# Global vars to hold loaded BM25 components
bm25 = None
bm25_corpus = None
bm25_meta = None

# Load all BM25 components (index, corpus, metadata)
def load_indexes():
    global bm25, bm25_corpus, bm25_meta

    with open(os.path.join(INDEX_DIR, "bm25.pkl"), "rb") as f:
        bm25 = pickle.load(f)

    with open(os.path.join(INDEX_DIR, "bm25_corpus.json"), "r") as f:
        bm25_corpus = json.load(f)

    with open(os.path.join(INDEX_DIR, "bm25_metadata.json"), "r") as f:
        bm25_meta = json.load(f)

# Perform top-k BM25 retrieval
def retrieve(query: str, k=5):
    if bm25 is None:
        load_indexes()

    # Tokenize the query
    query_tokens = query.strip().split()
    scores = bm25.get_scores(query_tokens)

    # Get top-k indices
    top_k_indices = np.argsort(scores)[::-1][:k]

    # Collect top-k hits
    hits = [{
        "score": float(scores[i]),
        "doc": bm25_corpus[i],
        "meta": bm25_meta[i],
        "method": "bm25"
    } for i in top_k_indices]

    return hits
