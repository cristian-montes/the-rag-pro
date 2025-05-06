#!/usr/bin/env python

import re
from load_mistral import load_mistral
from retrieval import (
    retrieve_bm25, retrieve_faiss,
    load_bm25_index, load_faiss_index
)
from corpus_loader.preprocess import preprocess
from corpus_loader.load_all_data import load_all_data
from corpus_loader.corpus_loader import main as corpus_loader_main

# Cache to avoid reloading resources
_cached_data = {}

# Constants
MAX_CONTEXT_TOKENS = 2000
GENERATION_MAX_TOKENS = 600
STOP_TOKENS = ["</s>", "###", "Answer:"]


def preprocess_query(query):
    """Lowercase and strip punctuation."""
    return re.sub(r'[^\w\s]', '', query.lower())


def select_best_result(bm25_results, faiss_results):
    """Select best result between BM25 and FAISS based on normalized scores."""
    best_bm25 = bm25_results[0]
    best_faiss = faiss_results[0]
    normalized_faiss_score = 1 / (1 + best_faiss['distance'])

    return (best_bm25, "BM25") if best_bm25['score'] > normalized_faiss_score else (best_faiss, "FAISS")


def combine_context_and_query(query, best_result, retrieval_method, max_tokens=MAX_CONTEXT_TOKENS):
    """Truncate context and format it with the query."""
    best_context = " ".join(best_result['document'].split()[:max_tokens])
    meta = best_result.get("metadata", {})

    combined = (
        f"Question: {query}\n"
        f"Context (from {retrieval_method}): {best_context}\n"
        "Answer:"
    )
    return combined, meta.get("source", "Unknown source"), meta.get("url", "not found"), meta.get("title", "N/A")


def load_cached_resources():
    """Load and cache model, index, and metadata."""
    if not _cached_data:
        corpus_loader_main()
        _cached_data.update({
            'model': load_mistral(),
            'bm25': None, 'bm25_corpus': None, 'bm25_metadata': None,
            'faiss_index': None, 'vectorizer': None, 'faiss_corpus': None, 'faiss_metadata': None,
        })

        # Load and unpack all indexes
        _cached_data['bm25'], _cached_data['bm25_corpus'], _cached_data['bm25_metadata'] = load_bm25_index()
        _cached_data['faiss_index'], _cached_data['vectorizer'], _cached_data['faiss_corpus'], _cached_data['faiss_metadata'] = load_faiss_index()
        _cached_data['preprocessed_corpus'] = preprocess(_cached_data['bm25_corpus'])

    return _cached_data


def main():
    resources = load_cached_resources()

    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break
        if not query or not re.search(r'\w', query):
            print("⚠️  Please enter a valid question with actual words.")
            continue

        cleaned_query = preprocess_query(query)
        print("\n🔍 Retrieving context...")

        bm25_results = retrieve_bm25(cleaned_query, resources['bm25'], resources['bm25_corpus'], resources['bm25_metadata'])
        faiss_results = retrieve_faiss(cleaned_query, resources['faiss_index'], resources['vectorizer'], resources['faiss_corpus'], resources['faiss_metadata'])

        best_result, method = select_best_result(bm25_results, faiss_results)
        combined_input, source, url, title = combine_context_and_query(query, best_result, method)

        response = resources['model'](
            combined_input,
            max_tokens=GENERATION_MAX_TOKENS,
            stop=STOP_TOKENS,
            echo=False
        )

        print(f"\n🧠 Model Response ({method} used):\n{response['choices'][0]['text'].strip()}")
        print(f"📚 Title: {title}\n🔗 URL: {url}\n")


if __name__ == "__main__":
    main()
