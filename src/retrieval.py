# retrieval.py

import json, os, pickle, faiss, numpy as np
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

INDEX_DIR = "index"

bm25 = None
bm25_corpus = None
bm25_meta = None

faiss_index = None
vectorizer = None
faiss_corpus = None
faiss_meta = None

def load_indexes():
    global bm25, bm25_corpus, bm25_meta
    global faiss_index, vectorizer, faiss_corpus, faiss_meta

    def _load(fname, loader):
        path = os.path.join(INDEX_DIR, fname)
        with open(path, 'rb') as f:
            return loader(f)

    bm25 = _load("bm25.pkl", pickle.load)
    bm25_corpus = json.load(open(os.path.join(INDEX_DIR, "bm25_corpus.json")))
    bm25_meta = json.load(open(os.path.join(INDEX_DIR, "bm25_metadata.json")))

    faiss_index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.idx"))
    vectorizer = _load("tfidf.pkl", pickle.load)
    faiss_corpus = json.load(open(os.path.join(INDEX_DIR, "faiss_corpus.json")))
    faiss_meta = json.load(open(os.path.join(INDEX_DIR, "faiss_metadata.json")))

def retrieve(query: str, k=4):
    if bm25 is None or faiss_index is None:
        load_indexes()

    # ── BM25 ──
    q_tok = query.split()
    bm25_scores = bm25.get_scores(q_tok)
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:k]
    bm25_hits = [{
        "score": float(bm25_scores[i]),
        "doc": bm25_corpus[i],
        "meta": bm25_meta[i],
        "method": "bm25"
    } for i in top_bm25_idx]

    # ── FAISS ──
    q_vec_sparse = vectorizer.transform([query])

    # Convert sparse matrix to dense float32 ndarray
    q_vec = q_vec_sparse.toarray().astype(np.float32)
    if q_vec.ndim == 1:
        q_vec = q_vec.reshape(1, -1)
    # print("DEBUG: reshaped q_vec.shape:", q_vec.shape)

    D, I = faiss_index.search(q_vec, k)
    faiss_hits = [{
        "score": float(1 / (1 + d + 1e-6)),  # convert distance→similarity
        "doc": faiss_corpus[i],
        "meta": faiss_meta[i],
        "method": "faiss"
    } for i, d in zip(I[0], D[0])]

    # Combine & sort by score
    return sorted(bm25_hits + faiss_hits, key=lambda x: -x["score"])[:k]

