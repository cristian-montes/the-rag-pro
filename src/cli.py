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

# 🔹 UPDATED: Function to select the best retrieval result
def select_best_result(bm25_results, bm25_scores, faiss_results, faiss_distances):
    """Dynamically selects the best result from BM25 and FAISS."""
    best_bm25_idx, best_bm25_score = bm25_results[0], bm25_scores[0]
    best_faiss_idx, best_faiss_distance = faiss_results[0], faiss_distances[0]
    
    # Normalize FAISS score for fair comparison
    if best_bm25_score > (1 / (best_faiss_distance + 1)):
        return best_bm25_idx, "BM25"
    else:
        return best_faiss_idx, "FAISS"

# 🔹 UPDATED: Only use the best context
def combine_context_and_query(query, best_result_idx, retrieval_method, corpus, tokenizer, max_tokens=512):
    """Combines the best retrieved context with the query while enforcing token limits."""
    best_context = corpus[best_result_idx]
    context_tokens = tokenizer.encode(best_context, add_special_tokens=False)
    
    if len(context_tokens) > max_tokens:
        context_tokens = context_tokens[:max_tokens]
    best_context = tokenizer.decode(context_tokens)
    
    return f"Question: {query}\nContext (from {retrieval_method}): {best_context}\nAnswer:"

def main():
    corpus_loader_main()
    model, tokenizer = load_mistral()
    bm25 = load_bm25_index()
    faiss_index, vectorizer = load_faiss_index()
    corpus = load_all_data()
    preprocessed_corpus = preprocess(corpus)
    
    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        
        cleaned_query = preprocess_query(query)
        print("\nRetrieving context...")
        bm25_results, bm25_scores = retrieve_bm25(cleaned_query, bm25, preprocessed_corpus)
        faiss_results, faiss_distances = retrieve_faiss(cleaned_query, faiss_index, vectorizer, preprocessed_corpus, top_k=5)
        
        # 🔹 UPDATED: Use the best retrieval result
        best_result_idx, retrieval_method = select_best_result(bm25_results, bm25_scores, faiss_results, faiss_distances)
        
        combined_input = combine_context_and_query(query, best_result_idx, retrieval_method, corpus, tokenizer)
        
        inputs = tokenizer(combined_input, return_tensors="pt").to("mps")
        outputs = model.generate(**inputs, max_new_tokens=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\nModel Response ({retrieval_method} used):\n{response}\n")

if __name__ == "__main__":
    main()




# #!/usr/bin/env python

# from load_mistral import load_mistral
# from retrieval import retrieve_bm25, retrieve_faiss, load_bm25_index, load_faiss_index
# from corpus_loader.preprocess import preprocess
# from corpus_loader.load_all_data import load_all_data
# from corpus_loader.corpus_loader import main as corpus_loader_main
# import re

# def preprocess_query(query):
#     """Preprocess the query to match the format of the preprocessed corpus"""
#     query = query.lower()  # Convert to lowercase
#     query = re.sub(r'[^\w\s]', '', query)  # Remove non-alphanumeric characters
#     return query

# def combine_context_and_query(query, bm25_results, faiss_results, corpus, preprocessed_corpus, top_k=3):
#     """Combine the retrieved context from BM25 and FAISS with the user's query"""
#     # Combine top results from both retrieval methods, removing duplicates
#     top_results = set(bm25_results[:top_k]) | set(faiss_results[:top_k])
#     context = "\n".join([corpus[idx] for idx in top_results])
    
#     # Combine the query with the context
#     combined_input = f"Question: {query}\nContext:\n{context}\nAnswer:"
#     return combined_input


# def main():
#     # Load the corpus,indexes
#     corpus_loader_main()
    
#     # Load the model and tokenizer
#     model, tokenizer = load_mistral()

#     # Load the BM25 and FAISS indices
#     bm25 = load_bm25_index()
#     faiss_index, vectorizer = load_faiss_index()

#     # Load your raw corpus data (scraped data from PDFs, CSVs, Wikipedia, NASA)
#     corpus = load_all_data()  # Or use corpus_loader if necessary

#     preprocessed_corpus = preprocess(corpus)  # Clean the corpus data

#     while True:
#         query = input("\nEnter your question (or type 'exit' to quit): ")
#         if query.lower() == "exit":
#             break

#         # Preprocess the user query before passing to the retrieval functions
#         cleaned_query = preprocess_query(query)

#         # Retrieve context from corpus using BM25 or FAISS
#         print("\nRetrieving context...")
#         bm25_results, bm25_scores = retrieve_bm25(cleaned_query, bm25, preprocessed_corpus)
#         faiss_results, faiss_distances = retrieve_faiss(cleaned_query, faiss_index, vectorizer, preprocessed_corpus, top_k=5)

#         # Combine top results from both retrieval methos
#         # top_resutls = set(bm25_results[:3]) | set(faiss_results[:3])
#         # context = "\n".join([corpus[idx] for idx in top_resutls])
#         # print(f"\nRetrieved Context:\n{context}")

#         combined_input = combine_context_and_query(query, bm25_results, faiss_results, corpus, preprocessed_corpus)
#         inputs = tokenizer(combined_input, return_tensors="pt").to("mps")
#         outputs = model.generate(**inputs, max_new_tokens=100)
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         # Pass context to the model
#         # inputs = tokenizer(query, return_tensors="pt").to("mps")
#         # outputs = model.generate(**inputs, max_new_tokens=100)
#         # response = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         # Display Model's response
#         print(f"\nModel Response:\n{response}\n")

# if __name__ == "__main__":
#     main()
