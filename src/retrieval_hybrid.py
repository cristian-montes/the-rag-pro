import json, os, pickle, faiss, numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

# Directory where indexes and metadata are stored
INDEX_DIR = "index"

# Global variables to hold loaded indexes and metadata
bm25 = None
bm25_corpus = None
bm25_meta = None

faiss_index = None
vectorizer = None
faiss_corpus = None
faiss_meta = None

# Loads all index and metadata files into global memory (only once)
def load_indexes():
    global bm25, bm25_corpus, bm25_meta
    global faiss_index, vectorizer, faiss_corpus, faiss_meta

    # Helper function to load binary objects (pickle, faiss)
    def _load(fname, loader):
        path = os.path.join(INDEX_DIR, fname)
        with open(path, 'rb') as f:
            return loader(f)

    # Load BM25 objects: model + corpus + metadata
    bm25 = _load("bm25.pkl", pickle.load)
    bm25_corpus = json.load(open(os.path.join(INDEX_DIR, "bm25_corpus.json")))
    bm25_meta = json.load(open(os.path.join(INDEX_DIR, "bm25_metadata.json")))

    # Load FAISS vector index, TF-IDF vectorizer, corpus, and metadata
    faiss_index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.idx"))
    vectorizer = _load("tfidf.pkl", pickle.load)
    faiss_corpus = json.load(open(os.path.join(INDEX_DIR, "faiss_corpus.json")))
    faiss_meta = json.load(open(os.path.join(INDEX_DIR, "faiss_metadata.json")))

# Core retrieval function using both BM25 and FAISS
def retrieve(query: str, k=4):
    # Load indexes on first run (lazy loading)
    if bm25 is None or faiss_index is None:
        load_indexes()

    # ── BM25 retrieval ──
    # Tokenize query and compute BM25 relevance scores
    q_tok = query.split()
    bm25_scores = bm25.get_scores(q_tok)

    # Get top-k results based on BM25 scores
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:k]
    bm25_hits = [{
        "score": float(bm25_scores[i]),
        "doc": bm25_corpus[i],
        "meta": bm25_meta[i],
        "method": "bm25"
    } for i in top_bm25_idx]

    # ── FAISS retrieval ──
    # Convert the query to a sparse TF-IDF vector
    q_vec_sparse = vectorizer.transform([query])

    # Convert to dense float32 format for FAISS compatibility
    q_vec = q_vec_sparse.toarray().astype(np.float32)
    if q_vec.ndim == 1:
        q_vec = q_vec.reshape(1, -1)

    # Perform similarity search with FAISS index
    D, I = faiss_index.search(q_vec, k)  # D = distances, I = indices

    # Convert distance to a similarity-like score and collect top-k hits
    faiss_hits = [{
        "score": float(1 / (1 + d + 1e-6)),  # transform distance to similarity
        "doc": faiss_corpus[i],
        "meta": faiss_meta[i],
        "method": "faiss"
    } for i, d in zip(I[0], D[0])]

    # Combine BM25 + FAISS hits and re-sort by descending score
    return sorted(bm25_hits + faiss_hits, key=lambda x: -x["score"])[:k]
