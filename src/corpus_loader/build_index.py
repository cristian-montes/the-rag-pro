import os
import json
import pickle
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from corpus_loader.load_all_data import load_all_data
from corpus_loader.preprocess import preprocess

INDEX_DIR = "index"
os.makedirs(INDEX_DIR, exist_ok=True)

def save_json(obj, name):
    with open(os.path.join(INDEX_DIR, name), "w") as f:
        json.dump(obj, f, indent=2)

def build():
    # Check if indexes already exist
    bm25_path = os.path.join(INDEX_DIR, "bm25.pkl")
    faiss_path = os.path.join(INDEX_DIR, "faiss.idx")

    if os.path.exists(bm25_path) and os.path.exists(faiss_path):
        print("Indexes already exist. Skipping build.")
        return

    raw_docs, raw_meta = load_all_data()
    # chunks, chunk_meta = preprocess(raw_docs)  # important: chunk-level!
    chunks, chunk_meta = preprocess(raw_docs, raw_meta, max_tokens=128, overlap=32)
    save_json(raw_docs, "raw_corpus.json")  # optional debugging
    save_json(raw_meta, "raw_metadata.json")

    # ── BM25 ───────────────────────────────────────────────────
    tokenized_chunks = [c.split() for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    save_json(chunks, "bm25_corpus.json")
    save_json(chunk_meta, "bm25_metadata.json")
    print("✅ BM25 built.")

    # ── FAISS (TF-IDF vectors) ─────────────────────────────────
    vectorizer = TfidfVectorizer()
    mat = vectorizer.fit_transform(chunks).astype("float32")
    index = faiss.IndexFlatL2(mat.shape[1])
    index.add(mat.toarray())
    faiss.write_index(index, faiss_path)
    with open(os.path.join(INDEX_DIR, "tfidf.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    save_json(chunks, "faiss_corpus.json")
    save_json(chunk_meta, "faiss_metadata.json")
    print("✅ FAISS built.")

if __name__ == "__main__":
    build()


