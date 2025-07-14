import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Paths
INDEX_DIR = "index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Globals
faiss_index = None
dense_corpus = None
dense_metadata = None
model = SentenceTransformer(EMBEDDING_MODEL)

# Load dense index and related data
def load_dense_index():
    global faiss_index, dense_corpus, dense_metadata
    if faiss_index is not None:
        return

    # Load FAISS index
    index_path = os.path.join(INDEX_DIR, "dense_index.faiss")
    faiss_index = faiss.read_index(index_path)

    # Load corpus and metadata
    with open(os.path.join(INDEX_DIR, "dense_corpus.json"), "r") as f:
        dense_corpus = json.load(f)
    with open(os.path.join(INDEX_DIR, "dense_metadata.json"), "r") as f:
        dense_metadata = json.load(f)

# Retrieve top-k similar chunks using dense retrieval
def retrieve(query: str, k=5):
    load_dense_index()

    # Encode query
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype("float32")


    # Normalize the query vector - ONLY COSINE SIM
    faiss.normalize_L2(query_vector)

    # Search index
    scores, indices = faiss_index.search(query_vector, k)

    # Format hits
    hits = []
    for score, idx in zip(scores[0], indices[0]):
        hits.append({
            "score": float(score),
            "doc": dense_corpus[idx],
            "meta": dense_metadata[idx],
            "method": "dense"
        })

    return hits
