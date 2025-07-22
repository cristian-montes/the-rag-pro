import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from corpus_preloader.load_all_data import load_all_data
from dense.dense_corpus_loader.preprocess import preprocess

INDEX_DIR = "index"
os.makedirs(INDEX_DIR, exist_ok=True)

def save_json(obj, filename):
    with open(os.path.join(INDEX_DIR, filename), "w") as f:
        json.dump(obj, f, indent=2)

def build():
    index_path = os.path.join(INDEX_DIR, "dense_index.faiss")
    if os.path.exists(index_path):
        print("✅ FAISS index already exists. Skipping build.")
        return

    print("📦 Loading raw data...")
    raw_docs, raw_meta = load_all_data()

    print("🧹 Preprocessing and chunking...")
    chunks, chunk_meta = preprocess(raw_docs, raw_meta, max_tokens=300, overlap=50)

    print("🤖 Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("🔢 Encoding chunks into embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("🔄 Normalizing embeddings for cosine similarity...")
    faiss.normalize_L2(embeddings)  #Essential for cosine similarity (magnituded)

    print("🧠 Building FAISS index with cosine similarity...")
    dim = embeddings.shape[1]
    #index = faiss.IndexFlatL2(dim) # Eucledean distance 
    index = faiss.IndexFlatIP(dim)  # Inner product works as cosine similarity after normalization
    index.add(embeddings)

    print("💾 Saving index and metadata...")
    faiss.write_index(index, index_path)
    save_json(chunks, "dense_corpus.json")
    save_json(chunk_meta, "dense_metadata.json")
    save_json(raw_docs, "raw_corpus.json")
    save_json(raw_meta, "raw_metadata.json")

    print("✅ Dense index built and saved.")

if __name__ == "__main__":
    build()

