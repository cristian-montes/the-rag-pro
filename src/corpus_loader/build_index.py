import os
import json
import pickle
from rank_bm25 import BM25Okapi
from corpus_loader.load_all_data import load_all_data
from corpus_loader.preprocess import preprocess

# Directory to store BM25 index and related files
INDEX_DIR = "index"
os.makedirs(INDEX_DIR, exist_ok=True)

# Utility function to save JSON data
def save_json(obj, filename):
    with open(os.path.join(INDEX_DIR, filename), "w") as f:
        json.dump(obj, f, indent=2)

# Builds BM25 index from corpus
def build():
    bm25_path = os.path.join(INDEX_DIR, "bm25.pkl")
    if os.path.exists(bm25_path):
        print("✅ BM25 index already exists. Skipping build.")
        return

    print("📦 Loading data...")
    raw_docs, raw_meta = load_all_data()

    print("🧹 Preprocessing and chunking...")
    chunks, chunk_meta = preprocess(
        raw_docs,
        raw_meta,
        max_tokens=300,
        overlap=50
    )

    print("🧠 Tokenizing chunks...")
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    print("💾 Saving index and metadata...")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    save_json(chunks, "bm25_corpus.json")
    save_json(chunk_meta, "bm25_metadata.json")
    save_json(raw_docs, "raw_corpus.json")
    save_json(raw_meta, "raw_metadata.json")

    print("✅ BM25 index built and saved.")

# Run if this file is executed directly
if __name__ == "__main__":
    build()
