#!/usr/bin/env python

from load_mistral import load_mistral
from retrieval import retrieve_bm25, retrieve_faiss, load_bm25_index, load_faiss_index
from corpus_loader.preprocess import preprocess
from corpus_loader.load_all_data import load_all_data
from corpus_loader.corpus_loader import main as corpus_loader_main
import re

def preprocess_query(query):
    """Preprocess the query to match the format of the preprocessed corpus"""
    query = query.lower()
    query = re.sub(r'[^\w\s]', '', query)
    return query

def select_best_result(bm25_results, bm25_scores, faiss_results, faiss_distances):
    """Dynamically selects the best result from BM25 and FAISS."""
    best_bm25_idx, best_bm25_score = bm25_results[0], bm25_scores[0]
    best_faiss_idx, best_faiss_distance = faiss_results[0], faiss_distances[0]
    
    if best_bm25_score > (1 / (best_faiss_distance + 1)):
        return best_bm25_idx, "BM25"
    else:
        return best_faiss_idx, "FAISS"

def combine_context_and_query(query, best_result_idx, retrieval_method, corpus, max_tokens=2000):
    """Combines the best retrieved context with the query."""
    best_context = corpus[best_result_idx]
    # (Optional) Limit context length to roughly fit in prompt
    if len(best_context.split()) > max_tokens:
        best_context = " ".join(best_context.split()[:max_tokens])
    
    return f"Question: {query}\nContext (from {retrieval_method}): {best_context}\nAnswer:"

def main():
    corpus_loader_main()
    model = load_mistral()
    bm25 = load_bm25_index()
    faiss_index, vectorizer = load_faiss_index()
    corpus = load_all_data()
    preprocessed_corpus = preprocess(corpus)
    
    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        #trims and checks for non-empty meaningful content
        if not query.strip() or not re.search(r'\w', query):
            print("⚠️  Please enter a valid question with actual words.")
            continue
        
        cleaned_query = preprocess_query(query)
        print("\nRetrieving context...")
        bm25_results, bm25_scores = retrieve_bm25(cleaned_query, bm25, preprocessed_corpus)
        faiss_results, faiss_distances = retrieve_faiss(cleaned_query, faiss_index, vectorizer, preprocessed_corpus, top_k=5)
        
        best_result_idx, retrieval_method = select_best_result(bm25_results, bm25_scores, faiss_results, faiss_distances)
        
        combined_input = combine_context_and_query(query, best_result_idx, retrieval_method, corpus)

        response = model(
            combined_input,
            max_tokens=100,
            stop=["</s>", "###", "Answer:"],
            echo=False
        )

        print(f"\nModel Response ({retrieval_method} used):\n{response['choices'][0]['text'].strip()}\n")

if __name__ == "__main__":
    main()
