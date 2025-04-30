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






# import os
# import json
# import pickle
# import faiss
# import numpy as np
# from rank_bm25 import BM25Okapi
# from sklearn.feature_extraction.text import TfidfVectorizer

# INDEX_DIR = "./index"

# # Ensure index directory exists
# os.makedirs(INDEX_DIR, exist_ok=True)


# def build_bm25_index(corpus, metadata):
#     """
#     Build and save a BM25 index along with metadata mapping.
#     :param corpus: List of document texts
#     :param metadata: List of metadata dicts corresponding to each document
#     """
#     # Tokenize corpus
#     tokenized_corpus = [doc.split() for doc in corpus]
#     bm25 = BM25Okapi(tokenized_corpus)

#     # Save BM25 index using pickle
#     bm25_path = os.path.join(INDEX_DIR, "bm25_index.pkl")
#     with open(bm25_path, 'wb') as f:
#         pickle.dump(bm25, f)
#     print(f"BM25 index saved at {bm25_path}.")

#     # Save metadata mapping for BM25
#     bm25_meta_path = os.path.join(INDEX_DIR, "bm25_metadata.json")
#     with open(bm25_meta_path, 'w') as mf:
#         json.dump(metadata, mf, indent=2)
#     print(f"BM25 metadata saved at {bm25_meta_path}.")


# def build_faiss_index(corpus, metadata):
#     """
#     Build and save a FAISS index along with metadata mapping.
#     :param corpus: List of document texts
#     :param metadata: List of metadata dicts corresponding to each document
#     """
#     # Convert texts into numeric vectors using TF-IDF
#     vectorizer = TfidfVectorizer()
#     matrix = vectorizer.fit_transform(corpus)

#     # Create FAISS index
#     index = faiss.IndexFlatL2(matrix.shape[1])  # L2 norm for similarity
#     index.add(matrix.toarray().astype('float32'))

#     # Save FAISS index
#     faiss_index_path = os.path.join(INDEX_DIR, "faiss_index.idx")
#     faiss.write_index(index, faiss_index_path)
#     print(f"FAISS index saved at {faiss_index_path}.")

#     # Save the TF-IDF vectorizer
#     vectorizer_path = os.path.join(INDEX_DIR, "tfidf_vectorizer.pkl")
#     with open(vectorizer_path, "wb") as f:
#         pickle.dump(vectorizer, f)
#     print(f"TF-IDF vectorizer saved at {vectorizer_path}.")

#     # Save metadata mapping for FAISS
#     faiss_meta_path = os.path.join(INDEX_DIR, "faiss_metadata.json")
#     with open(faiss_meta_path, 'w') as mf:
#         json.dump(metadata, mf, indent=2)
#     print(f"FAISS metadata saved at {faiss_meta_path}.")

