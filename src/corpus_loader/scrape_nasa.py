import os
import json
import requests
from bs4 import BeautifulSoup

NASA_URL = "https://www.nasa.gov/missions/"
CACHE_FILE = "data/nasa/nasa_missions.json"

def scrape_nasa():
    nasa_texts = []
    try:
        response = requests.get(NASA_URL)
        soup = BeautifulSoup(response.text, 'html.parser')
        for item in soup.find_all('p'):
            text = item.get_text().strip()
            if text:
                nasa_texts.append(text)
    except Exception as e:
        print(f"Failed to scrape NASA data: {e}")
    return nasa_texts


#Loads cache or if cache does not exists scrapes fresh data to load a new cache
def get_nasa_data():
    if os.path.exists(CACHE_FILE):
        print("Loading cached NASA data")
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    
    else:
        print("Fetching new NASA data...")
        data = scrape_nasa()

        #Checking data/nasa directories exists
        os.makedirs("data/nasa", exist_ok=True)

        with open(CACHE_FILE, "w") as f:
            json.dump(data, f, indent=4)
        
        return data


if __name__ == "__main__":
    nasa_data = get_nasa_data()
    print(f"loaded{len(nasa_data[:3])}") #Preview first three 3 articles

    for i, text in enumerate(nasa_data[:3]):
        print(f"\nArticle {i + 1}:\n{text[:300]}...")
        print("=" * 40)
