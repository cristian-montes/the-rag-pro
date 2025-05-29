import os
import json
import fitz 
from .download_pdfs import download_pdfs

METADATA_FILE = "data/pdfs/metadata.json"

def load_saved_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return []

def find_metadata_for_file(filename, metadata_list):
    for item in metadata_list:
        if item.get("filename") == filename:
            return item
    return {
        "title": "Unknown Title",
        "description": "No description available.",
        "url": None,
        "source": "Unknown source",
        "filename": filename
    }

def load_pdfs(path):
    download_pdfs()
    texts = []
    metadata = []
    saved_metadata = load_saved_metadata()

    for file in os.listdir(path):
        if file.endswith(".pdf"):
            file_path = os.path.join(path, file)
            try:
                with fitz.open(file_path) as doc:
                    text = ""
                    for page in doc:
                        text += page.get_text()

                    texts.append(text)

                    meta = find_metadata_for_file(file, saved_metadata)
                    meta["filepath"] = file_path
                    metadata.append(meta)

            except Exception as e:
                print(f"❌ Failed to load {file}: {e}")

    return texts, metadata

if __name__ == "__main__":
    pdf_dir = 'data/pdfs'
    pdf_texts, pdf_meta = load_pdfs(pdf_dir)
    print(f"✅ Loaded {len(pdf_texts)} PDF files.")
    print("📌 Sample metadata:", pdf_meta[0] if pdf_meta else "None")
