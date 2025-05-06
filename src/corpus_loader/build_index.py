import os
import json
import pickle
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

INDEX_DIR = "./index"
os.makedirs(INDEX_DIR, exist_ok=True)


def build_bm25_index(corpus, metadata):
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(os.path.join(INDEX_DIR, "bm25_index.pkl"), 'wb') as f:
        pickle.dump(bm25, f)
    with open(os.path.join(INDEX_DIR, "bm25_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(INDEX_DIR, "bm25_corpus.json"), 'w') as f:
        json.dump(corpus, f, indent=2)

    print("✅ BM25 index, metadata, and corpus saved.")


def build_faiss_index(corpus, metadata):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)

    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix.toarray().astype('float32'))

    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss_index.idx"))
    with open(os.path.join(INDEX_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(INDEX_DIR, "faiss_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(INDEX_DIR, "faiss_corpus.json"), 'w') as f:
        json.dump(corpus, f, indent=2)

    print("✅ FAISS index, vectorizer, metadata, and corpus saved.")


def main():
    # Replace with actual corpus/metadata loading logic if needed
    corpus = [
        "The cat sat on the mat.",
        "The dog barked at the moon.",
        "Artificial Intelligence is transforming the world.",
        "Space exploration has advanced rapidly in recent years.",
        "Quantum computing is a breakthrough in technology."
    ]

    metadata = [
        {"source": "file1.txt"},
        {"source": "file2.txt"},
        {"source": "file3.txt"},
        {"source": "file4.txt"},
        {"source": "file5.txt"}
    ]

    build_bm25_index(corpus, metadata)
    build_faiss_index(corpus, metadata)


if __name__ == "__main__":
    main()

