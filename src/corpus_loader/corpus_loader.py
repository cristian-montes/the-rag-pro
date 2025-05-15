from .load_all_data import load_all_data
from .preprocess import preprocess
from .build_index import build_bm25_index, build_faiss_index

def main():
    # Load all raw data (PDFs, CSVs, Wikipedia, NASA)
    print("Loading raw data...")
    corpus, metadata = load_all_data()  # Get both content and metadata

    # Preprocess the data (content only)
    print("Preprocessing data...")
    cleaned_corpus = preprocess(corpus)

    print("Metadata has been loaded and can be used or saved.")

    # Build BM25 index
    print("Building BM25 index...")
    build_bm25_index(cleaned_corpus,metadata)

    # Build FAISS index
    print("Building FAISS index...")
    build_faiss_index(cleaned_corpus,metadata)

if __name__ == "__main__":
    main()
