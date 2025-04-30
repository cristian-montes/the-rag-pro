#!/usr/bin/env python

from load_mistral import load_mistral
from retrieval import retrieve_bm25, retrieve_faiss, load_bm25_index, load_faiss_index
from corpus_loader.preprocess import preprocess
from corpus_loader.load_all_data import load_all_data
from corpus_loader.corpus_loader import main as corpus_loader_main
import re

# Cache to avoid reloading every time
_cached_data = {}

def preprocess_query(query):
    """Preprocess the query to match the format of the preprocessed corpus"""
    query = query.lower()
    query = re.sub(r'[^\w\s]', '', query)
    return query

def select_best_result(bm25_results, faiss_results):
    """Dynamically selects the best result from BM25 and FAISS using normalized FAISS score."""
    best_bm25 = bm25_results[0]
    best_faiss = faiss_results[0]

    # Normalize FAISS score to match BM25 scale (1 / (1 + distance))
    normalized_faiss_score = 1 / (1 + best_faiss['distance'])

    if best_bm25['score'] > normalized_faiss_score:
        return best_bm25, "BM25"
    else:
        return best_faiss, "FAISS"

def combine_context_and_query(query, best_result, retrieval_method, max_tokens=2000):
    """Combines the best retrieved context with the query."""
    best_context = best_result['document']
    if len(best_context.split()) > max_tokens:
        best_context = " ".join(best_context.split()[:max_tokens])

    
    source = best_result.get("metadata", {}).get("source", "Unknown source")
    url = best_result.get("metadata", {}).get("url", "not found")
    title = best_result.get("metadata", {}).get("title", "N/A")

    combined = f"Question: {query}\nContext (from {retrieval_method}): {best_context}\nAnswer:"
    return combined, source, url, title

def load_cached_resources():
    if not _cached_data:
        corpus_loader_main()
        _cached_data['model'] = load_mistral()
        _cached_data['bm25'], _cached_data['bm25_metadata'] = load_bm25_index()
        _cached_data['faiss_index'], _cached_data['vectorizer'], _cached_data['faiss_metadata'] = load_faiss_index()
        corpus = load_all_data()
        _cached_data['preprocessed_corpus'] = preprocess(corpus)
    return _cached_data

def main():
    data = load_cached_resources()

    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        if not query.strip() or not re.search(r'\w', query):
            print("⚠️  Please enter a valid question with actual words.")
            continue

        cleaned_query = preprocess_query(query)
        print("\nRetrieving context...")

        bm25_results = retrieve_bm25(cleaned_query, data['bm25'], data['preprocessed_corpus'], data['bm25_metadata'])
        faiss_results = retrieve_faiss(cleaned_query, data['faiss_index'], data['vectorizer'], data['preprocessed_corpus'], data['faiss_metadata'])

        best_result, retrieval_method = select_best_result(bm25_results, faiss_results)
        combined_input, source, url, title = combine_context_and_query(query, best_result, retrieval_method)

        response = data['model'](
            combined_input,
            max_tokens=100,
            stop=["</s>", "###", "Answer:"],
            echo=False
        )
        
        print(f"\nModel Response ({retrieval_method} used):\n{response['choices'][0]['text'].strip()}")
        # print(f"📚 Source: {source}\n")
        print(f"📚 [{title}](url)-{source}\n")

if __name__ == "__main__":
    main()
