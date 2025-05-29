"""
Text cleaning + chunking that keeps accurate LLama token counts and
emits rich, per-chunk metadata for citations.
"""
import re, itertools, tiktoken

# Minimal and fast custom stopword list (expand if needed)
CUSTOM_STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "for", "to", "from", "by", "with",
    "is", "are", "was", "were", "be", "been", "of", "that", "this", "it", "as", "but",
    "if", "then", "so", "not", "no", "yes", "do", "does", "did", "doing", "have", "has", "had"
}

ENCODER = tiktoken.get_encoding("cl100k_base")  # close to LLama 7-B counts

def clean(text: str, remove_stop=True) -> str:
    text = re.sub(r"[^\w\s]", " ", text.lower())  # remove punctuation, lowercase
    text = re.sub(r"\s+", " ", text).strip()      # normalize whitespace
    if remove_stop:
        words = text.split()
        words = [w for w in words if w not in CUSTOM_STOPWORDS]
        return " ".join(words)
    return text

def chunk(text: str, max_tokens=128, overlap=32):
    words = text.split()
    ptr = 0
    while ptr < len(words):
        chunk_words = words[ptr:ptr + max_tokens]
        # guarantee ≤max_tokens real LLama tokens
        while len(ENCODER.encode(" ".join(chunk_words))) > max_tokens:
            chunk_words = chunk_words[:-1]
        yield " ".join(chunk_words)
        ptr += max_tokens - overlap

# def preprocess(docs, *, max_tokens=128, overlap=32):
#     """Flatten, clean & chunk. Return (chunks, metadata_for_chunks)."""
#     chunks, meta = [], []
#     for doc_id, doc in enumerate(itertools.chain.from_iterable(
#             d if isinstance(d, list) else [d] for d in docs)):
#         cleaned = clean(doc)
#         for ck_id, ck in enumerate(chunk(cleaned, max_tokens, overlap)):
#             chunks.append(ck)
#             meta.append({
#                 "doc_id":   doc_id,
#                 "chunk_id": ck_id,           # local chunk index in document
#                 "tokens":   len(ENCODER.encode(ck))
#             })
#     return chunks, meta

def preprocess(docs, meta_in, *, max_tokens=128, overlap=32):
    """Flatten, clean & chunk. Return (chunks, metadata_for_chunks)."""
    chunks, meta_out = [], []
    print('in the PREPROCESS')
    for doc_id, doc in enumerate(itertools.chain.from_iterable(
            d if isinstance(d, list) else [d] for d in docs)):
        cleaned = clean(doc)
        source_meta = meta_in[doc_id]
        for ck_id, ck in enumerate(chunk(cleaned, max_tokens, overlap)):
            chunk_meta = {
                "doc_id": doc_id,
                "chunk_id": ck_id,
                "tokens": len(ENCODER.encode(ck)),
                **source_meta  # merge original metadata here
            }
            chunks.append(ck)
            meta_out.append(chunk_meta)
    return chunks, meta_out

