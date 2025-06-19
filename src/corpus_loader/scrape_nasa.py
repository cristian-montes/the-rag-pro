import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager  # Automatically installs the correct ChromeDriver
from bs4 import BeautifulSoup  # For HTML parsing
import json

# Constants for scraping and caching
NASA_URL = "https://www.nasa.gov/missions/"
CACHE_FILE = "data/nasa/nasa_missions.json"       # Path to store scraped text content
METADATA_FILE = "data/nasa/metadata.json"         # Path to store metadata associated with the text

def scrape_nasa_selenium():
    nasa_texts = []   # Will hold the scraped mission text
    metadata = []     # Will hold metadata for each text block

    # Setup headless Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # No UI/browser window
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")

    # Launch Chrome using WebDriver Manager
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        driver.get(NASA_URL)              # Visit the missions page
        driver.implicitly_wait(5)         # Allow time for page load

        # Parse the page source using BeautifulSoup
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Try selecting mission content using a known class
        mission_divs = soup.select("div.mission-card")
        if not mission_divs:
            mission_divs = soup.find_all("p")  # Fallback if no mission cards found

        # Extract text and generate metadata for each mission card
        for idx, div in enumerate(mission_divs):
            text = div.get_text(separator="\n", strip=True)
            if text:
                nasa_texts.append(text)
                metadata.append({
                    "source": "NASA",
                    "url": NASA_URL,
                    "block_index": idx  # Index for ordering or tracing
                })

    except Exception as e:
        print(f"Failed to scrape NASA data: {e}")

    finally:
        driver.quit()  # Cleanly close browser session

    return nasa_texts, metadata  # Return content and its metadata

def get_nasa_data():
    # If previously scraped data is cached, load and return it
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
        return data, metadata
    else:
        # Otherwise, scrape fresh data from NASA site
        data, metadata = scrape_nasa_selenium()

        # Ensure output directory exists
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)

        # Save results to cache for future runs
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f, indent=4)
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=4)

        return data, metadata  # Return newly scraped data

if __name__ == "__main__":
    nasa_data, metadata = get_nasa_data()
    print(f"Loaded {len(nasa_data)} entries with metadata.")
