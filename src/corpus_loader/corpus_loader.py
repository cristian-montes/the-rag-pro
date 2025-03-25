# corpus_loader/corpus_loader.py
from load_pdfs import load_pdfs
from load_csv import load_csv
# from load_wikipedia import load_wikipedia
from load_wikipedia import get_wikipedia_data
# from scrape_nasa import scrape_nasa
from scrape_nasa import get_nasa_data
from preprocess import preprocess
from build_index import build_bm25_index, build_faiss_index

def main():
    print("Loading PDFs...")
    pdfs = load_pdfs('data/pdfs')

    print("Loading CSV data...")
    csv_data = load_csv('data/csvs')

    print("Loading Wikipedia pages...")
    titles = ['Mars', 'Apollo_program', 'SpaceX', 'Hubble_Space_Telescope']
    wiki_data = get_wikipedia_data(titles)

    print("Scraping NASA data...")
    nasa_data = get_nasa_data()

    print("Preprocessing data...")
    corpus = preprocess(pdfs + csv_data + wiki_data + nasa_data)

    print("Building BM25 index...")
    build_bm25_index(corpus)

    print("Building FAISS index...")
    build_faiss_index(corpus)

if __name__ == "__main__":
    main()




#  USING LOAD ALL DATA FOR CLEANER LOOK AND REUSABILITY
# from data_loader import load_all_data  # Import load_all_data function
# from preprocess import preprocess
# from build_index import build_bm25_index, build_faiss_index

# def main():
#     # Load all raw data (PDFs, CSVs, Wikipedia, NASA)
#     print("Loading raw data...")
#     corpus = load_all_data()

#     # Preprocess the data
#     print("Preprocessing data...")
#     cleaned_corpus = preprocess(corpus)

#     # Build BM25 index
#     print("Building BM25 index...")
#     build_bm25_index(cleaned_corpus)

#     # Build FAISS index
#     print("Building FAISS index...")
#     build_faiss_index(cleaned_corpus)

# if __name__ == "__main__":
#     main()
