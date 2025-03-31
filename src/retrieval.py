import os
import pickle
import faiss
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

INDEX_DIR = "./index"

def load_bm25_index():
    bm25_path = os.path.join(INDEX_DIR, "bm25_index.pkl")
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    print("BM25 index loaded.")
    return bm25

def load_faiss_index():
    faiss_index_path = os.path.join(INDEX_DIR, "faiss_index.idx")
    vectorizer_path = os.path.join(INDEX_DIR, "tfidf_vectorizer.pkl")

    # Load FAISS index
    index = faiss.read_index(faiss_index_path)
    
    # Load TF-IDF vectorizer
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    print("FAISS index and TF-IDF vectorizer loaded.")
    return index, vectorizer

def retrieve_bm25(query, bm25, corpus):
    """Retrieve documents using BM25"""
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = np.argsort(scores)[::-1]  # Sort in descending order
    return ranked_indices, scores

def retrieve_faiss(query, index, vectorizer, corpus, top_k=5):
    """Retrieve top K similar documents using FAISS"""
    query_vector = vectorizer.transform([query]).toarray().astype('float32')
    distances, indices = index.search(query_vector, top_k)
    
    return indices[0], distances[0]  # Return top-k results

def main():
    # Load stored indexes
    bm25 = load_bm25_index()
    faiss_index, vectorizer = load_faiss_index()

    # Sample corpus (ensure you load your real corpus here)
    corpus = [
        "The cat sat on the mat.",
        "The dog barked at the moon.",
        "Artificial Intelligence is transforming the world.",
        "Space exploration has advanced rapidly in recent years.",
        "Quantum computing is a breakthrough in technology."
    ]

    # Query input
    query = "space and technology"

    # Retrieve results from BM25
    bm25_results, bm25_scores = retrieve_bm25(query, bm25, corpus)
    print("\nBM25 Top Results:")
    for idx in bm25_results[:5]:
        print(f"Score: {bm25_scores[idx]:.4f} | Doc: {corpus[idx]}")

    # Retrieve results from FAISS
    faiss_results, faiss_distances = retrieve_faiss(query, faiss_index, vectorizer, corpus, top_k=5)
    print("\nFAISS Top Results:")
    for idx, distance in zip(faiss_results, faiss_distances):
        print(f"Distance: {distance:.4f} | Doc: {corpus[idx]}")

if __name__ == "__main__":
    main()
