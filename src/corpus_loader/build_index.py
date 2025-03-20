# corpus_loader/build_index.py
from rank_bm25 import BM25Okapi
import faiss
import numpy as np

def build_bm25_index(corpus):
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    with open('../index/bm25_index.pkl', 'wb') as f:
        import pickle
        pickle.dump(bm25, f)
    print("BM25 index saved.")

def build_faiss_index(corpus):
    # Convert texts into numeric vectors using simple TF-IDF encoding
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix.toarray().astype('float32'))

    faiss.write_index(index, "../index/faiss_index.idx")
    print("FAISS index saved.")
