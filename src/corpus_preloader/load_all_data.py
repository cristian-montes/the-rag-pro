import os
from .load_pdfs import load_pdfs              # Loads PDFs from disk
from .future_scripts_impl.scrape_nasa import get_nasa_data        # Scrapes/loads NASA data
from .future_scripts_impl.load_wikipedia import get_wikipedia_data  # Fetches Wikipedia content

# Constants for directories and sources
PDF_DIR = "data/pdfs"
CSV_DIR = "data/csvs"
WIKIPEDIA_TITLES = ['Mars', 'Apollo_program', 'SpaceX', 'Hubble_Space_Telescope']
INCLUDE_NASA_DATA = False  # Toggle NASA scraping on/off

# Loader for PDFs
def load_pdfs_data():
    """Load PDF data with metadata."""
    print("Loading PDFs...")
    return load_pdfs(PDF_DIR)  # returns (docs, metadata)

# Loader for Wikipedia articles
def load_wikipedia_data():
    """Load Wikipedia data with metadata."""
    print("Loading Wikipedia...")
    return get_wikipedia_data(WIKIPEDIA_TITLES)

# Loader for NASA content
def load_nasa_data():
    """Load NASA scraped data with metadata."""
    print("Loading NASA scraped content...")
    return get_nasa_data()

# Aggregator: load all sources and combine their outputs
def load_all_data():
    """
    Load all datasets: PDFs, CSVs, Wikipedia, and optionally NASA.
    Returns:
        Tuple[List[str], List[dict]]: corpus texts and their metadata
    """
    corpus = []     # stores raw document text
    metadata = []   # stores per-document metadata

    loaders = [load_pdfs_data]  # base loaders
    if INCLUDE_NASA_DATA:                            # conditionally add NASA
        loaders.append(load_nasa_data)

    # Execute each loader, extending global corpus + metadata
    for loader in loaders:
        docs, meta = loader()
        corpus.extend(docs)
        metadata.extend(meta)

    return corpus, metadata

# Run this file directly to debug loading process
if __name__ == "__main__":
    corpus, metadata = load_all_data()
    print(f"✅ Loaded {len(corpus)} documents with metadata.")
    print(f"📎 Example metadata:\n{metadata[0] if metadata else 'No metadata found.'}")
