import wikipediaapi
import json
import os

# User agent header (used for proper and respectful access to Wikipedia's API)
USER_AGENT = "the_rag_pro/1.0 (cristian.montes.bluestaq@gmail.com)"

# Initialize the Wikipedia API client for English language using the custom user agent
wiki_wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language='en')

# Paths for saving downloaded Wikipedia data and associated metadata
CACHE_FILE = "data/wikipedia/wikipedia_data.json"
METADATA_FILE = "data/wikipedia/metadata.json"

def fetch_wikipedia_articles(titles):
    articles = []  # Stores paragraph-level text
    metadata = []  # Stores metadata about each paragraph

    for title in titles:
        page = wiki_wiki.page(title)  # Fetch the Wikipedia page
        if page.exists():  # Only process if the page exists
            paragraphs = page.text.split('\n\n')  # Split text into paragraphs
            for idx, para in enumerate(paragraphs):
                if para.strip():  # Ignore empty sections
                    articles.append(para.strip())
                    metadata.append({
                        "source": "Wikipedia",
                        "title": title,
                        "paragraph": idx,        # Paragraph index for traceability
                        "url": page.fullurl      # Direct link to the Wikipedia page
                    })
        else:
            print(f"Warning: The Wikipedia page for '{title}' was not found.")

    return articles, metadata

def get_wikipedia_data(titles):
    # If cached data exists, load and return it
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            articles = json.load(f)
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
        return articles, metadata

    # Otherwise, fetch fresh data and save it
    else:
        articles, metadata = fetch_wikipedia_articles(titles)
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)

        # Save articles and metadata to disk
        with open(CACHE_FILE, "w") as f:
            json.dump(articles, f, indent=4)
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=4)

        return articles, metadata

if __name__ == "__main__":
    titles = ['Mars', 'Apollo_program', 'SpaceX', 'Hubble_Space_Telescope']
    data, metadata = get_wikipedia_data(titles)
    print(f"Loaded {len(data)} paragraphs with metadata.")
