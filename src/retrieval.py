import os
import json
import pickle
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

INDEX_DIR = "./index"

# Ensure index directory exists
os.makedirs(INDEX_DIR, exist_ok=True)

def load_bm25_index():
    bm25_path = os.path.join(INDEX_DIR, "bm25_index.pkl")
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    print("BM25 index loaded.")

    bm25_meta_path = os.path.join(INDEX_DIR, "bm25_metadata.json")
    with open(bm25_meta_path, "r") as mf:
        metadata = json.load(mf)
    print("BM25 metadata loaded.")

    bm25_corpus_path = os.path.join(INDEX_DIR, "bm25_corpus.json")
    with open(bm25_corpus_path, "r") as f:
        corpus = json.load(f)
    print("BM25 corpus loaded.")

    return bm25, corpus, metadata


def load_faiss_index():
    faiss_index_path = os.path.join(INDEX_DIR, "faiss_index.idx")
    vectorizer_path = os.path.join(INDEX_DIR, "tfidf_vectorizer.pkl")
    faiss_meta_path = os.path.join(INDEX_DIR, "faiss_metadata.json")
    faiss_corpus_path = os.path.join(INDEX_DIR, "faiss_corpus.json")

    index = faiss.read_index(faiss_index_path)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(faiss_meta_path, "r") as mf:
        metadata = json.load(mf)
    with open(faiss_corpus_path, "r") as f:
        corpus = json.load(f)

    print("FAISS index, TF-IDF vectorizer, metadata, and corpus loaded.")
    return index, vectorizer, corpus, metadata

def retrieve_bm25(query, bm25, corpus, metadata, top_k=5):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = np.argsort(scores)[::-1]
    results = []
    for idx in ranked_indices[:top_k]:
        results.append({
            "score": float(scores[idx]),
            "document": corpus[idx],
            "metadata": metadata[idx]
        })
    print(metadata[idx])    
    return results

def retrieve_faiss(query, index, vectorizer, corpus, metadata, top_k=5):
    query_vector = vectorizer.transform([query]).toarray().astype('float32')
    distances, indices = index.search(query_vector, top_k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "distance": float(dist),
            "document": corpus[idx],
            "metadata": metadata[idx]
        })
    print(metadata[idx]) 
    return results

def main():
    # Load BM25 index and associated metadata
    bm25, bm25_metadata = load_bm25_index()

    # Load FAISS index, TF-IDF vectorizer, and metadata
    faiss_index, vectorizer, faiss_metadata = load_faiss_index()

    # Load the real corpus used during indexing
    bm25_corpus_path = os.path.join(INDEX_DIR, "bm25_corpus.json")
    faiss_corpus_path = os.path.join(INDEX_DIR, "faiss_corpus.json")

    if os.path.exists(bm25_corpus_path):
        with open(bm25_corpus_path, "r") as f:
            bm25_corpus = json.load(f)
    else:
        print("BM25 corpus file not found.")
        return

    if os.path.exists(faiss_corpus_path):
        with open(faiss_corpus_path, "r") as f:
            faiss_corpus = json.load(f)
    else:
        print("FAISS corpus file not found.")
        return

    # Define a test query
    query = "space and technology"

    # Retrieve using BM25
    bm25_results = retrieve_bm25(query, bm25, bm25_corpus, bm25_metadata)
    print("\nBM25 Top Results:")
    for result in bm25_results:
        print(f"Score: {result['score']:.4f} | Doc: {result['document']} | Source: {result['metadata'].get('source', 'N/A')}")

    # Retrieve using FAISS
    faiss_results = retrieve_faiss(query, faiss_index, vectorizer, faiss_corpus, faiss_metadata)
    print("\nFAISS Top Results:")
    for result in faiss_results:
        print(f"Distance: {result['distance']:.4f} | Doc: {result['document']} | Source: {result['metadata'].get('source', 'N/A')}")


if __name__ == "__main__":
    main()

