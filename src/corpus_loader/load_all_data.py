import os
import glob
import csv
from .load_pdfs import load_pdfs
from .load_csv import load_csv
from .scrape_nasa import get_nasa_data
from .load_wikipedia import get_wikipedia_data

PDF_DIR = "data/pdfs"
CSV_DIR = "data/csvs"
WIKIPEDIA_TITLES = ['Mars', 'Apollo_program', 'SpaceX', 'Hubble_Space_Telescope']
NASA_DATA = True  # Set to True if you want to scrape NASA data

def load_pdfs_data():
    """Load PDF data from a specified directory"""
    print("Loading PDFs...")
    pdf_data = load_pdfs(PDF_DIR)
    return pdf_data

def load_csv_data():
    """Load CSV data from a specified directory"""
    print("Loading CSV data...")
    csv_data = load_csv(CSV_DIR)
    return csv_data

def load_wikipedia_data():
    """Load Wikipedia data for specific titles"""
    print("Loading Wikipedia data...")
    wiki_data = get_wikipedia_data(WIKIPEDIA_TITLES)
    return wiki_data

def load_nasa_data():
    """Load NASA scraped data"""
    print("Loading NASA data...")
    nasa_data = get_nasa_data()
    return nasa_data

def load_all_data():
    """Load all data sources (PDFs, CSVs, Wikipedia, NASA)"""
    corpus = []

    # Load the raw data
    corpus.extend(load_pdfs_data())  # Add PDF data
    corpus.extend(load_csv_data())   # Add CSV data
    corpus.extend(load_wikipedia_data())  # Add Wikipedia data
    
    if NASA_DATA:
        corpus.extend(load_nasa_data())

    return corpus

if __name__ == "__main__":
    # For testing purposes, load all data
    all_data = load_all_data()
    print(f"Loaded {len(all_data)} documents.")
