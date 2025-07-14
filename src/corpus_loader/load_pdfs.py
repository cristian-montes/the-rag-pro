import os
import json
import fitz  # PyMuPDF: used to open and extract text from PDFs

# Call to download PDFs beforehand (ensures required files are present)
from .static_download_pdfs import download_selected_pdfs

# Path to saved metadata that describes each PDF file (manual or pre-created)
METADATA_FILE = "data/pdfs/metadata.json"

# Load the metadata JSON file if it exists
def load_saved_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return []

# Helper: find matching metadata entry for a given filename
def find_metadata_for_file(filename, metadata_list):
    for item in metadata_list:
        if item.get("filename") == filename:
            return item
    # If no metadata found, fallback to generic info
    return {
        "title": "Unknown Title",
        "description": "No description available.",
        "url": None,
        "source": "Unknown source",
        "filename": filename
    }

# Main loader function to extract text and metadata from all PDFs in the given path
def load_pdfs(path):
    download_selected_pdfs()  # Make sure PDFs are downloaded before processing

    texts = []     # Raw text of each PDF
    metadata = []  # Metadata for each PDF
    saved_metadata = load_saved_metadata()

    for file in os.listdir(path):
        if file.endswith(".pdf"):  # Only process PDF files
            file_path = os.path.join(path, file)
            try:
                with fitz.open(file_path) as doc:  # Open PDF using PyMuPDF
                    text = ""
                    for page in doc:
                        text += page.get_text()  # Extract text page by page

                    texts.append(text)  # Save entire document text

                    meta = find_metadata_for_file(file, saved_metadata)
                    meta["filepath"] = file_path  # Store file location in metadata
                    metadata.append(meta)

            except Exception as e:
                print(f"❌ Failed to load {file}: {e}")

    return texts, metadata  # Return list of doc texts and matching metadata

# Test/run manually
if __name__ == "__main__":
    pdf_dir = 'data/pdfs'
    pdf_texts, pdf_meta = load_pdfs(pdf_dir)
    print(f"✅ Loaded {len(pdf_texts)} PDF files.")
    print("📌 Sample metadata:", pdf_meta[0] if pdf_meta else "None")
