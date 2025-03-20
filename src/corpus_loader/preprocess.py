# corpus_loader/preprocess.py
import re

def preprocess(texts):
    cleaned_texts = []
    for text in texts:
        # Lowercase and remove non-alphanumeric characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        cleaned_texts.append(text)
    return cleaned_texts
