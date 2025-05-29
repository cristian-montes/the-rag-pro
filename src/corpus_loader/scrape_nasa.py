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

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        driver.get(NASA_URL)
        driver.implicitly_wait(5)

        soup = BeautifulSoup(driver.page_source, "html.parser")

        mission_divs = soup.select("div.mission-card")  # Adjust selector as needed
        if not mission_divs:
            mission_divs = soup.find_all("p")  # fallback

        for idx, div in enumerate(mission_divs):
            text = div.get_text(separator="\n", strip=True)
            if text:
                nasa_texts.append(text)
                metadata.append({
                    "source": "NASA",
                    "url": NASA_URL,
                    "block_index": idx
                })

    except Exception as e:
        print(f"Failed to scrape NASA data: {e}")

    finally:
        driver.quit()

    return nasa_texts, metadata

def get_nasa_data():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
        return data, metadata
    else:
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
