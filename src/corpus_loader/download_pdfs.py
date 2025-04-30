import os
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Constants
BASE_URL = "https://www.nasa.gov/ebooks/"
PDF_DIR = "data/pdfs"
METADATA_FILE = "data/pdfs/metadata.json"

# Ensure directory exists
os.makedirs(PDF_DIR, exist_ok=True)

def load_metadata():
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            print(f"⚠️ Warning: Could not parse {METADATA_FILE}. Resetting metadata.")
            return []
    return []

def save_metadata(metadata):
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

def fetch_ebook_links():
    """Fetch all overview page URLs from NASA eBooks."""
    response = requests.get(BASE_URL)
    soup = BeautifulSoup(response.text, "html.parser")
    ebook_links = []

    # Find all <a> tags that act as 'Overview' buttons
    for a in soup.find_all("a", class_="button-primary"):
        if a.get_text(strip=True).lower() == "overview":
            overview_url = a.get("href")
            if overview_url:
                ebook_links.append(urljoin(BASE_URL, overview_url))

    print(f"🔍 Found {len(ebook_links)} overview pages")
    return ebook_links

def extract_pdf_links_and_metadata(overview_url):
    response = requests.get(overview_url)
    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.find("h1").text.strip() if soup.find("h1") else "Unknown Title"
    description_tag = soup.find("meta", attrs={"name": "description"})
    description = description_tag["content"] if description_tag else "No description available."

    pdf_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        parsed = urlparse(href)

        # Check if the path ends with .pdf (ignore query params)
        if parsed.path.lower().endswith(".pdf"):
            full_url = urljoin(overview_url, href)
            pdf_links.append(full_url)

    return title, description, pdf_links


def download_pdf(pdf_url):
    # Strip query params from filename
    parsed_url = urlparse(pdf_url)
    filename = os.path.basename(parsed_url.path)  # this removes query string
    filepath = os.path.join(PDF_DIR, filename)

    if os.path.exists(filepath):
        print(f"Already downloaded (file check): {filename}")
        return filename

    print(f"Downloading: {filename}")
    response = requests.get(pdf_url, stream=True)
    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    return filename

def main():
    ebook_links = fetch_ebook_links()
    print(f"Found {len(ebook_links)} overview links.")

    # Limit to 50 for local development
    max_ebooks = 50
    ebook_links = ebook_links[:max_ebooks]

    metadata = load_metadata()
    downloaded_urls = {entry["pdf_url"] for entry in metadata}

    for overview_url in ebook_links:
        title, description, pdf_links = extract_pdf_links_and_metadata(overview_url)

        for pdf_url in pdf_links:
            if pdf_url in downloaded_urls:
                print(f"Already downloaded (metadata): {pdf_url}")
                continue

            filename = download_pdf(pdf_url)
            metadata.append({
                "title": title,
                "description": description,
                "pdf_url": pdf_url,
                "filename": filename,
                "source": overview_url
            })
            downloaded_urls.add(pdf_url)

    save_metadata(metadata)

    print(f"✅ Completed. Total PDFs in metadata: {len(metadata)}")

if __name__ == "__main__":
    main()
