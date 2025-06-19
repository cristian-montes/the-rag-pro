import os
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm  # For displaying progress bars in loops

BASE_URL = "https://www.nasa.gov/ebooks/"
PDF_DIR = "data/pdfs"
METADATA_FILE = "data/pdfs/metadata.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NASA-Scraper/1.0)"}
MIN_VALID_SIZE = 1024  # Minimum file size (1 KB) to be considered valid

os.makedirs(PDF_DIR, exist_ok=True)

def load_metadata():
    """Load previously saved metadata if available."""
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
    """Save metadata to file."""
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)


def fetch_ebook_links():
    """Fetch all overview page URLs from NASA eBooks landing page."""
    try:
        response = requests.get(BASE_URL, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Failed to fetch ebook list: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    ebook_links = []

    for a in soup.find_all("a", class_="button-primary"):
        if a.get_text(strip=True).lower() == "overview":
            overview_url = a.get("href")
            if overview_url:
                ebook_links.append(urljoin(BASE_URL, overview_url))

    print(f"🔍 Found {len(ebook_links)} overview pages")
    return ebook_links

def extract_pdf_links_and_metadata(overview_url):
    """Parse an overview page to extract PDF links and basic metadata."""
    try:
        response = requests.get(overview_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Failed to fetch overview page {overview_url}: {e}")
        return "Unknown Title", "No description available.", []

    soup = BeautifulSoup(response.text, "html.parser")

    # Try to find the book title and description
    title = soup.find("h1").text.strip() if soup.find("h1") else "Unknown Title"
    description_tag = soup.find("meta", attrs={"name": "description"})
    description = description_tag["content"] if description_tag else "No description available."

    # Extract any links to PDF files
    pdf_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        parsed = urlparse(href)
        if parsed.path.lower().endswith(".pdf"):
            full_url = urljoin(overview_url, href)
            pdf_links.append(full_url)

    return title, description, pdf_links

def download_pdf(pdf_url):
    """Download a PDF if not already downloaded and validate its size."""
    parsed_url = urlparse(pdf_url)
    filename = os.path.basename(parsed_url.path)
    filepath = os.path.join(PDF_DIR, filename)

    if os.path.exists(filepath):
        if os.path.getsize(filepath) >= MIN_VALID_SIZE:
            print(f"Already downloaded (file check): {filename}")
            return filename
        else:
            print(f"⚠️ Found tiny file. Re-downloading: {filename}")
            os.remove(filepath)

    try:
        print(f"⬇️ Downloading: {filename}")
        response = requests.get(pdf_url, stream=True, headers=HEADERS, timeout=30)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        if os.path.getsize(filepath) < MIN_VALID_SIZE:
            print(f"⚠️ Skipping too-small file: {filename}")
            os.remove(filepath)
            return None

        return filename

    except requests.RequestException as e:
        print(f"❌ Failed to download {pdf_url}: {e}")
        return None

def download_pdfs(max_ebooks=10):
    """Main function to fetch and download a set number of NASA eBook PDFs."""
    ebook_links = fetch_ebook_links()
    if not ebook_links:
        return

    # Limit the number of eBooks processed (optional cap)
    ebook_links = ebook_links[:max_ebooks]

    # Load existing metadata to avoid redundant downloads
    metadata = load_metadata()
    downloaded_urls = {entry["url"] for entry in metadata}

    # Process each overview page and its PDFs
    for overview_url in tqdm(ebook_links, desc="Processing eBooks"):
        title, description, pdf_links = extract_pdf_links_and_metadata(overview_url)

        for pdf_url in pdf_links:
            if pdf_url in downloaded_urls:
                print(f"Already downloaded (metadata): {pdf_url}")
                continue

            filename = download_pdf(pdf_url)
            if not filename:
                continue

            metadata.append({
                "title": title,
                "description": description,
                "url": pdf_url,
                "filename": filename,
                "source": overview_url
            })
            downloaded_urls.add(pdf_url)

    save_metadata(metadata)
    print(f"✅ Completed. Total PDFs in metadata: {len(metadata)}")

if __name__ == "__main__":
    download_pdfs()
