import os
import json
import requests
from urllib.parse import urlparse
from tqdm import tqdm

PDF_DIR = "data/pdfs"
METADATA_FILE = "data/pdfs/metadata.json"
MIN_VALID_SIZE = 1024  # Minimum file size (1 KB) to be considered valid
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NASA-Scraper/1.0)"}

#  PDFs URLs to be mostly text
HARDCODED_PDF_URLS = [
    "https://www.nasa.gov/wp-content/uploads/2015/01/Archaeology_Anthropology_and_Interstellar_Communication_TAGGED.pdf?emrc=9e061d",
    "https://www.nasa.gov/wp-content/uploads/2025/02/governing-the-moon-sp-2024-4559-ebook.pdf?emrc=686dfb5c7fb7d",
    "https://www.nasa.gov/wp-content/uploads/2015/04/621513main_rocketspeoplevolume4-ebook.pdf?emrc=4d370f"
]

os.makedirs(PDF_DIR, exist_ok=True)

def load_metadata():
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r") as f:
                content = f.read().strip()
                return json.loads(content) if content else []
        except Exception:
            print("⚠️ Could not parse metadata. Starting fresh.")
    return []

def save_metadata(metadata):
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

def download_pdf(pdf_url):
    filename = os.path.basename(urlparse(pdf_url).path)
    filepath = os.path.join(PDF_DIR, filename)

    if os.path.exists(filepath) and os.path.getsize(filepath) >= MIN_VALID_SIZE:
        print(f"✅ Already exists: {filename}")
        return filename

    try:
        print(f"⬇️ Downloading: {filename}")
        response = requests.get(pdf_url, stream=True, headers=HEADERS, timeout=30)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(1024):
                if chunk:
                    f.write(chunk)

        if os.path.getsize(filepath) < MIN_VALID_SIZE:
            print(f"⚠️ Skipping too-small file: {filename}")
            os.remove(filepath)
            return None

        return filename
    except Exception as e:
        print(f"❌ Failed: {pdf_url} — {e}")
        return None

def download_selected_pdfs():
    metadata = load_metadata()
    already_downloaded = {entry["url"] for entry in metadata}

    for url in tqdm(HARDCODED_PDF_URLS, desc="Downloading PDFs"):
        if url in already_downloaded:
            print(f"🟡 Skipping (already in metadata): {url}")
            continue

        filename = download_pdf(url)
        if not filename:
            continue

        metadata.append({
            "title": filename.replace(".pdf", "").replace("_", " "),
            "description": "Hardcoded PDF assumed to be mostly text.",
            "url": url,
            "filename": filename,
            "source": "manual"
        })

    save_metadata(metadata)
    print(f"\n✅ Done. Total PDFs: {len(metadata)}")

if __name__ == "__main__":
    download_selected_pdfs()
