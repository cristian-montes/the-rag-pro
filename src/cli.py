#!/usr/bin/env python

from load_mistral import load_mistral
from retrieval import retrieve_bm25, retrieve_faiss, load_bm25_index, load_faiss_index
from corpus_loader import preprocess  # Import your preprocess function
from corpus_loader import load_all_data # If you want to load data, otherwise use corpus_loader
import re

def preprocess_query(query):
    """Preprocess the query to match the format of the preprocessed corpus"""
    query = query.lower()  # Convert to lowercase
    query = re.sub(r'[^\w\s]', '', query)  # Remove non-alphanumeric characters
    return query

def main():
    # Load the model and tokenizer
    model, tokenizer = load_mistral()

    # Load the BM25 and FAISS indices
    bm25 = load_bm25_index()
    faiss_index, vectorizer = load_faiss_index()

    # Load your raw corpus data (scraped data from PDFs, CSVs, Wikipedia, NASA)
    corpus = load_all_data()  # Or use corpus_loader if necessary

    # Preprocess the corpus data
    preprocessed_corpus = preprocess(corpus)  # Clean the corpus data

    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        # Preprocess the user query before passing to the retrieval functions
        cleaned_query = preprocess_query(query)

        # Retrieve context from corpus using BM25 or FAISS
        print("\nRetrieving context...")
        bm25_results, bm25_scores = retrieve_bm25(cleaned_query, bm25, preprocessed_corpus)
        faiss_results, faiss_distances = retrieve_faiss(cleaned_query, faiss_index, vectorizer, preprocessed_corpus, top_k=5)
        
        # Combine or choose best context
        context = ""
        for idx in bm25_results[:5]:
            context += preprocessed_corpus[idx] + "\n"
        
        print(f"\nRetrieved context (BM25 top results):\n{context[:500]}...")  # Show the first 500 chars of the context

        # Pass context to the model
        inputs = tokenizer(query, return_tensors="pt").to("mps")
        outputs = model.generate(**inputs, max_new_tokens=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Display Model's response
        print(f"\nModel Response:\n{response}\n")

if __name__ == "__main__":
    main()

# from load_mistral import load_mistral
# from retrieval import retrieve_text

# def main():
#     model, tokenizer = load_mistral()

#     while True:
#         query = input("\nEnter your question (or type 'exit' to quit): ")
#         if query.lower() == "exit":
#             break

#         # Retrive context from corpus
#         context = retrieve_text(query)
#         print(f"\nRetrived context: {context}")

#         # Pass context to the model
#         inputs = tokenizer(query, return_tensors="pt").to("mps")
#         outputs = model.generate(**inputs, max_new_tokens=100)
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         #Display Result
#         print(f"\nModel Response:\n{response}\n")

# if __name__ == "__main__":
#     main()