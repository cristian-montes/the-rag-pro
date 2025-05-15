import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import json

NASA_URL = "https://www.nasa.gov/missions/"
CACHE_FILE = "data/nasa/nasa_missions.json"
METADATA_FILE = "data/nasa/metadata.json"

def scrape_nasa_selenium():
    nasa_texts = []
    metadata = []

    # Configure Selenium (Headless Mode for Faster Scraping)
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # Use newer headless mode for stability
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")  # Ensure full page loads

    # Automatically download & manage ChromeDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        driver.get(NASA_URL)
        driver.implicitly_wait(5)  # Wait for JavaScript to load

        # Parse page source with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Extract text from <p> and <div> elements
        for item in soup.find_all(["p", "div"]):
            text = item.get_text().strip()
            if text:
                nasa_texts.append(text)
                # Add mission metadata (for example, using section headers or some identifiers)
                metadata.append({
                    "source": "NASA",
                    "url": NASA_URL
                })

    except Exception as e:
        print(f"Failed to scrape NASA data: {e}")

    finally:
        driver.quit()  # Close the browser session

    return nasa_texts, metadata

def get_nasa_data():
    if os.path.exists(CACHE_FILE):
        print("Loading cached NASA data...")
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
        return data, metadata
    else:
        print("Fetching new NASA data...")
        data, metadata = scrape_nasa_selenium()

        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f, indent=4)

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=4)

        return data, metadata

if __name__ == "__main__":
    nasa_data, metadata = get_nasa_data()
    print(f"Loaded {len(nasa_data)} entries with metadata.")
