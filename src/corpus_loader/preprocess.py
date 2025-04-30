import re

def flatten_texts(texts):
    """Recursively flattens a list of strings or list of lists of strings."""
    for item in texts:
        if isinstance(item, list):
            yield from flatten_texts(item)
        elif isinstance(item, str):
            yield item

def preprocess(texts):
    """
    Clean and flatten a list of text documents for indexing and retrieval.

    :param texts: List[str] or List[List[str]] raw document texts
    :return: List[str] cleaned and flattened document texts
    """
    cleaned_texts = []
    for text in flatten_texts(texts):
        t = text.lower()
        t = re.sub(r'[^\w\s]', '', t)
        t = re.sub(r'\s+', ' ', t).strip()
        cleaned_texts.append(t)
    return cleaned_texts

if __name__ == "__main__":
    samples = [["Hello, WORLD!!!", "Second line"], "This is   a test..."]
    print(preprocess(samples))





# # corpus_loader/preprocess.py
# import re

# def preprocess(texts):
#     cleaned_texts = []
#     for text in texts:
#         # Lowercase and remove non-alphanumeric characters
#         text = text.lower()
#         text = re.sub(r'[^\w\s]', '', text)
#         cleaned_texts.append(text)
#     return cleaned_texts
