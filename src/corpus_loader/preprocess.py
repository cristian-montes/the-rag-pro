"""
Text cleaning + chunking that keeps accurate LLaMA token counts and
emits rich, per-chunk metadata for citations.
"""

import re, itertools, tiktoken

# ── Custom stopword list ──
# Used to remove common uninformative words from the text
CUSTOM_STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "for", "to", "from", "by", "with",
    "is", "are", "was", "were", "be", "been", "of", "that", "this", "it", "as", "but",
    "if", "then", "so", "not", "no", "yes", "do", "does", "did", "doing", "have", "has", "had"
}

# ── Token encoder ──
# Approximates LLaMA's tokenization for accurate token length limits
ENCODER = tiktoken.get_encoding("cl100k_base")  # similar to LLaMA 7B tokenizer

# ── Clean function ──
# Lowercases text, removes punctuation, and optionally removes stopwords
def clean(text: str, remove_stop=True) -> str:
    text = re.sub(r"[^\w\s]", " ", text.lower())  # Remove punctuation, lowercase
    text = re.sub(r"\s+", " ", text).strip()      # Collapse extra spaces
    if remove_stop:
        words = text.split()
        words = [w for w in words if w not in CUSTOM_STOPWORDS]
        return " ".join(words)
    return text

# ── Chunking function ──
# Splits cleaned text into overlapping chunks under the max token limit
def chunk(text: str, max_tokens=128, overlap=32):
    words = text.split()
    ptr = 0
    while ptr < len(words):
        chunk_words = words[ptr:ptr + max_tokens]
        # Ensure chunk has no more than `max_tokens` real tokens
        while len(ENCODER.encode(" ".join(chunk_words))) > max_tokens:
            chunk_words = chunk_words[:-1]
        yield " ".join(chunk_words)
        ptr += max_tokens - overlap  # slide window forward with overlap

# ── Main preprocessing function ──
# Cleans and chunks each document; returns text chunks and detailed metadata
def preprocess(docs, meta_in, *, max_tokens=128, overlap=32):
    """
    Flatten, clean, and chunk documents.
    Returns:
        - chunks: list of text chunks
        - meta_out: list of metadata dicts for each chunk
    """
    chunks, meta_out = [], []
    print('in the PREPROCESS')

    # Iterate over each document (handles lists-of-lists)
    for doc_id, doc in enumerate(itertools.chain.from_iterable(
            d if isinstance(d, list) else [d] for d in docs)):

        cleaned = clean(doc)  # Clean the text
        source_meta = meta_in[doc_id]  # Original document metadata

        # Chunk the cleaned text and generate metadata for each chunk
        for ck_id, ck in enumerate(chunk(cleaned, max_tokens, overlap)):
            chunk_meta = {
                "doc_id": doc_id,
                "chunk_id": ck_id,
                "tokens": len(ENCODER.encode(ck)),  # token count of the chunk
                **source_meta  # inherit and merge original metadata
            }
            chunks.append(ck)
            meta_out.append(chunk_meta)

    return chunks, meta_out
