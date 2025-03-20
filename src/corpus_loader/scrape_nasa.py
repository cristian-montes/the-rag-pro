import requests
from bs4 import BeautifulSoup

NASA_URL = "https://www.nasa.gov/missions/"

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

if __name__ == "__main__":
    nasa_url = 'https://www.nasa.gov/missions'
    nasa_texts = scrape_nasa(nasa_url)
    print(f"Loaded {len(nasa_texts)} NASA mission articles.")