import wikipediaapi
import json
import os

USER_AGENT = "the_rag_pro/1.0 (cristian.montes.bluestaq@gmail.com)"
wiki_wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language='en')

CACHE_FILE = "data/wikipedia/wikipedia_data.json"
METADATA_FILE = "data/wikipedia/metadata.json"

def fetch_wikipedia_articles(titles):
    articles = []
    metadata = []
    for title in titles:
        page = wiki_wiki.page(title)
        if page.exists():
            paragraphs = page.text.split('\n\n')
            for idx, para in enumerate(paragraphs):
                if para.strip():
                    articles.append(para.strip())
                    metadata.append({
                        "source": "Wikipedia",
                        "title": title,
                        "paragraph": idx,
                        "url": page.fullurl
                    })
        else:
            print(f"Warning: The Wikipedia page for '{title}' was not found.")
    return articles, metadata

def get_wikipedia_data(titles):
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            articles = json.load(f)
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
        return articles, metadata
    else:
        articles, metadata = fetch_wikipedia_articles(titles)
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(articles, f, indent=4)
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=4)
        return articles, metadata

if __name__ == "__main__":
    titles = ['Mars', 'Apollo_program', 'SpaceX', 'Hubble_Space_Telescope']
    data, metadata = get_wikipedia_data(titles)
    print(f"Loaded {len(data)} paragraphs with metadata.")
