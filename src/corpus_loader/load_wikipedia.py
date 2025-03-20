import wikipediaapi

def load_wikipedia(pages=['Mars', 'Apollo_program', 'SpaceX', 'Hubble_Space_Telescope']):
  wiki - wikipediaapi.Wikipedia('en')
  wiki_texts = []
  for page in pages:
    try:
      p = wiki.page(page)
      if p.exists():
        wiki_texts.append(p.text)
      else:
        print(f"Page '{page}', does not exist.")
    except Exception as e:
      print(f"Failed to load Wiki page '{page}': {e}")
  return wiki_texts

if __name__ == "__main__":
    topics = ['Artemis program', 'James Webb Space Telescope', 'Mars Rover', 'SpaceX']
    wiki_texts = load_wikipedia(topics)
    print(f"Loaded {len(wiki_texts)} Wikipedia pages.")