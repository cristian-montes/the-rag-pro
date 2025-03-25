from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import json
import os

NASA_URL = "https://www.nasa.gov/missions/"
CACHE_FILE = "data/nasa/nasa_missions.json"

def scrape_nasa_selenium():
    nasa_texts = []

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

    except Exception as e:
        print(f"Failed to scrape NASA data: {e}")

    finally:
        driver.quit()  # Close the browser session

    return nasa_texts

def get_nasa_data():
    if os.path.exists(CACHE_FILE):
        print("Loading cached NASA data")
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)

        # Ensure data is returned as a list
        return list(data) if isinstance(data, dict) else data
    else:
        print("Fetching new NASA data...")
        data = scrape_nasa_selenium()
        
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f, indent=4)

        return data


if __name__ == "__main__":
    nasa_data = get_nasa_data()
    print(f"Loaded {len(nasa_data)} entries")

    for i, text in enumerate(nasa_data[:3]):
        print(f"\nArticle {i + 1}:\n{text[:300]}...")
        print("=" * 40)
