import os
from .load_pdfs import load_pdfs
from .load_csv import load_csv
from .scrape_nasa import get_nasa_data
from .load_wikipedia import get_wikipedia_data

PDF_DIR = "data/pdfs"
CSV_DIR = "data/csvs"
WIKIPEDIA_TITLES = ['Mars', 'Apollo_program', 'SpaceX', 'Hubble_Space_Telescope']
INCLUDE_NASA_DATA = True

def load_pdfs_data():
    """Load PDF data with metadata."""
    print("Loading PDFs...")
    return load_pdfs(PDF_DIR)

def load_csv_data():
    """Load CSV data with metadata."""
    print("Loading CSVs...")
    return load_csv(CSV_DIR)

def load_wikipedia_data():
    """Load Wikipedia data with metadata."""
    print("Loading Wikipedia...")
    return get_wikipedia_data(WIKIPEDIA_TITLES)

def load_nasa_data():
    """Load NASA scraped data with metadata."""
    print("Loading NASA scraped content...")
    return get_nasa_data()

def load_all_data():
    """
    Load all datasets: PDFs, CSVs, Wikipedia, and optionally NASA.
    Returns:
        Tuple[List[str], List[dict]]: corpus texts and their metadata
    """
    corpus = []
    metadata = []

    loaders = [load_pdfs_data, load_csv_data, load_wikipedia_data]
    if INCLUDE_NASA_DATA:
        loaders.append(load_nasa_data)

    for loader in loaders:
        docs, meta = loader()
        corpus.extend(docs)
        metadata.extend(meta)
        
    print(f"Corpus size: {len(corpus)}")
    print(f"Metadata size: {len(metadata)}")
    return corpus, metadata

if __name__ == "__main__":
    corpus, metadata = load_all_data()
    print(f"✅ Loaded {len(corpus)} documents with metadata.")
    print(f"📎 Example metadata:\n{metadata[0] if metadata else 'No metadata found.'}")
