import wikipediaapi
import json
import os

wiki_wiki = wikipediaapi.Wikipedia('en')
CACHE_FILE = "data/wikipedia/wikipedia_data.json"

# Function to fetch Wikipedia articles for the given list of titles
def fetch_wikipedia_articles(titles):
    articles = {}
    for title in titles:
        page = wiki_wiki.page(title)
        if page.exists():
            articles[title] = page.text
        else:
            print(f"Warning: The Wikipedia page for '{title}' was not found.")
    return articles

# Function to load Wikipedia data, either from cache or by fetching it
def get_wikipedia_data(titles):
    # Check if the cache file exists
    if os.path.exists(CACHE_FILE):
        print("Loading cached Wikipedia data...")
        # Load and return the cached data
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    else:
        print("Fetching new Wikipedia data...")
        # Fetch the data and cache it for future use
        data = fetch_wikipedia_articles(titles)

        # Ensure the 'data' folder exists
        if not os.path.exists('data/wikipedia'):
            os.makedirs('data/wikipedia')

        # Save the fetched data to the cache file
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f, indent=4)

        return data
    

# Example usage:
if __name__ == "__main__":
    # List of Wikipedia article titles you want to fetch
    titles = ['Mars', 'Apollo_program', 'SpaceX', 'Hubble_Space_Telescope']

    # Get Wikipedia data (either from cache or by fetching)
    wikipedia_data = get_wikipedia_data(titles)

    # Optionally, print the fetched data
    for title, text in wikipedia_data.items():
        print(f"Article: {title}")
        print(f"Text: {text[:300]}...")  # Print first 300 characters for preview
        print("=" * 40)






# def load_wikipedia(pages=['Mars', 'Apollo_program', 'SpaceX', 'Hubble_Space_Telescope']):

#   wiki = wikipediaapi.Wikipedia('en')

#   wiki_texts = []
#   for page in pages:
#     try:
#       p = wiki.page(page)
#       if p.exists():
#         wiki_texts.append(p.text)
#       else:
#         print(f"Page '{page}', does not exist.")
#     except Exception as e:
#       print(f"Failed to load Wiki page '{page}': {e}")
#   return wiki_texts

# if __name__ == "__main__":
#     topics = ['Artemis program', 'James Webb Space Telescope', 'Mars Rover', 'SpaceX']
#     wiki_texts = load_wikipedia(topics)
#     print(f"Loaded {len(wiki_texts)} Wikipedia pages.")