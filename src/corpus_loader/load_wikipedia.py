import wikipediaapi
import json
import os

# Define the user agent (replace 'your-email@example.com' with a real contact email)
USER_AGENT = "the_rag_pro/1.0 (cristian.montes.bluestaq@gmail.com)"

# Initialize Wikipedia API with user agent
wiki_wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language='en')

CACHE_FILE = "data/wikipedia/wikipedia_data.json"
METADATA_FILE = "data/wikipedia/metadata.json"

def fetch_wikipedia_articles(titles):
    articles = {}
    metadata = []
    for title in titles:
        page = wiki_wiki.page(title)
        if page.exists():
            articles[title] = page.text
            metadata.append({
                "source": "Wikipedia",
                "title": title,
                "url": page.fullurl
            })
        else:
            print(f"Warning: The Wikipedia page for '{title}' was not found.")
    return articles, metadata

def get_wikipedia_data(titles):
    if os.path.exists(CACHE_FILE):
        print("Loading cached Wikipedia data...")
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
        return list(data.values()), metadata
    else:
        print("Fetching new Wikipedia data...")
        articles, metadata = fetch_wikipedia_articles(titles)

        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)

        with open(CACHE_FILE, "w") as f:
            json.dump(articles, f, indent=4)
        
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=4)

        return list(articles.values()), metadata

# Example usage
if __name__ == "__main__":
    titles = ['Mars', 'Apollo_program', 'SpaceX', 'Hubble_Space_Telescope']
    wikipedia_data, metadata = get_wikipedia_data(titles)

    print(f"Loaded {len(wikipedia_data)} articles with metadata.")




# import wikipediaapi
# import json
# import os

# # Define the user agent (replace 'your-email@example.com' with a real contact email)
# USER_AGENT = "the_rag_pro/1.0 (cristian.montes.bluestaq@gmail.com)"

# # Initialize Wikipedia API with user agent
# wiki_wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language='en')

# CACHE_FILE = "data/wikipedia/wikipedia_data.json"

# # Function to fetch Wikipedia articles for the given list of titles
# def fetch_wikipedia_articles(titles):
#     articles = {}
#     for title in titles:
#         page = wiki_wiki.page(title)
#         if page.exists():
#             articles[title] = page.text
#         else:
#             print(f"Warning: The Wikipedia page for '{title}' was not found.")
#     return articles

# # Function to load Wikipedia data, either from cache or by fetching it
# def get_wikipedia_data(titles):
#     if os.path.exists(CACHE_FILE):
#         print("Loading cached Wikipedia data...")
#         with open(CACHE_FILE, "r") as f:
#             data = json.load(f)
#         return list(data.values()) if isinstance(data, dict) else data
#     else:
#         print("Fetching new Wikipedia data...")
#         data = fetch_wikipedia_articles(titles)

#         # Ensure the directory exists before saving
#         os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)

#         with open(CACHE_FILE, "w") as f:
#             json.dump(data, f, indent=4)

#         return list(data.values())

# # Example usage
# if __name__ == "__main__":
#     titles = ['Mars', 'Apollo_program', 'SpaceX', 'Hubble_Space_Telescope']
#     wikipedia_data = get_wikipedia_data(titles)

#     for title, text in wikipedia_data.items():
#         print(f"Article: {title}")
#         print(f"Text: {text[:300]}...")  # Print first 300 characters
#         print("=" * 40)
