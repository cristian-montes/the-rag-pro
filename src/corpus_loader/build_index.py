import os
import pickle
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

INDEX_DIR = "../index"

# Ensure index directory exists
os.makedirs(INDEX_DIR, exist_ok=True)

def build_bm25_index(corpus):
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # Save BM25 index using pickle
    bm25_path = os.path.join(INDEX_DIR, "bm25_index.pkl")
    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25, f)
    
    print(f"BM25 index saved at {bm25_path}.")

def build_faiss_index(corpus):
    # Convert texts into numeric vectors using TF-IDF
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)

    # Create FAISS index
    index = faiss.IndexFlatL2(matrix.shape[1])  # L2 norm for similarity
    index.add(matrix.toarray().astype('float32'))

    # Save FAISS index
    faiss_index_path = os.path.join(INDEX_DIR, "faiss_index.idx")
    faiss.write_index(index, faiss_index_path)

    # Save the TF-IDF vectorizer (so we can transform queries later)
    vectorizer_path = os.path.join(INDEX_DIR, "tfidf_vectorizer.pkl")
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"FAISS index saved at {faiss_index_path}.")
    print(f"TF-IDF vectorizer saved at {vectorizer_path}.")
