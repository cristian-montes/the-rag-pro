# corpus_loader/corpus_loader.py
from load_pdfs import load_pdfs
from load_csv import load_csv
from load_wikipedia import load_wikipedia
from scrape_nasa import scrape_nasa
from preprocess import preprocess
from build_index import build_bm25_index, build_faiss_index

def main():
    print("Loading PDFs...")
    pdfs = load_pdfs()

    print("Loading CSV data...")
    csv_data = load_csv()

    print("Loading Wikipedia pages...")
    wiki_data = load_wikipedia(["NASA", "Apollo 11", "Mars"])

    print("Scraping NASA data...")
    nasa_data = scrape_nasa()

    print("Preprocessing data...")
    corpus = preprocess(pdfs + csv_data + wiki_data + nasa_data)

    print("Building BM25 index...")
    build_bm25_index(corpus)

    print("Building FAISS index...")
    build_faiss_index(corpus)

if __name__ == "__main__":
    main()
