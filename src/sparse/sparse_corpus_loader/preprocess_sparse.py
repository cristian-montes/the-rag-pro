"""
Text cleaning + chunking for sparse retrieval (e.g., BM25).
Removes stopwords, lemmatizes, and splits into overlapping word chunks.
Emits per-chunk metadata for citations.
"""

import html
import unicodedata
import re
import itertools
import spacy

# Load spaCy model once (download with: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

def clean(text: str) -> str:
    # HTML decode, normalize, clean punctuation, etc.
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[\u00ad\u200b\u200e\u200f]', '', text)
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = text.lower()

    # Split into safe-size chunks (below spaCy's max length)
    chunk_size = 50000  # 50K characters per batch (safe)
    lemmas = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        doc = nlp(chunk)
        chunk_lemmas = [t.lemma_ for t in doc if t.is_alpha and not t.is_stop]
        lemmas.extend(chunk_lemmas)

    return " ".join(lemmas).strip()



def chunk(text: str, max_words=270, overlap=40):
    words = text.split()
    ptr = 0
    while ptr < len(words):
        chunk_words = words[ptr:ptr + max_words]
        yield " ".join(chunk_words)
        ptr += max_words - overlap

def preprocess(docs, meta_in, *, max_words=270, overlap=40):
    """
    Flatten, clean, lemmatize, and chunk documents.
    Returns:
        - chunks: list of text chunks
        - meta_out: list of metadata dicts for each chunk
    """
    chunks, meta_out = [], []
    print("🧹 Preprocessing documents...")

    for doc_id, doc in enumerate(itertools.chain.from_iterable(
            d if isinstance(d, list) else [d] for d in docs)):

        cleaned = clean(doc)
        source_meta = meta_in[doc_id]

        for ck_id, ck in enumerate(chunk(cleaned, max_words, overlap)):
            chunk_meta = {
                "doc_id": doc_id,
                "chunk_id": ck_id,
                "words": len(ck.split()),
                **source_meta
            }
            chunks.append(ck)
            meta_out.append(chunk_meta)

    return chunks, meta_out











# DOES NOT HAVE LEMANTIZATION


# """
# Text cleaning + chunking that keeps accurate LLaMA token counts and
# emits rich, per-chunk metadata for citations.
# SLIDING WINDOW - CHUNKING STRATEGY .... MAX TOKENS AND OVERLAPS TO NOT MISS CONTEX
# """

# import re, itertools, tiktoken
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# # ── Token encoder ──
# # Approximates LLaMA's tokenization for accurate token length limits
# ENCODER = tiktoken.get_encoding("cl100k_base")  # similar to LLaMA 7B tokenizer

# def clean(text: str) -> str:
#     words = text.lower().split()
#     return ' '.join([word for word in words if word not in ENGLISH_STOP_WORDS])

# # ── Chunking function ──
# # Splits cleaned text into overlapping chunks under the max token limit
# def chunk(text: str, max_tokens=400, overlap=40):
#     words = text.split()
#     ptr = 0
#     while ptr < len(words):
#         chunk_words = words[ptr:ptr + max_tokens]
#         # Ensure chunk has no more than `max_tokens` real tokens
#         while len(ENCODER.encode(" ".join(chunk_words))) > max_tokens:
#             chunk_words = chunk_words[:-1]
#         yield " ".join(chunk_words)
#         ptr += max_tokens - overlap  # slide window forward with overlap

# # ── Main preprocessing function ──
# # Cleans and chunks each document; returns text chunks and detailed metadata
# def preprocess(docs, meta_in, *, max_tokens=160, overlap=40):
#     """
#     Flatten, clean, and chunk documents.
#     Returns:
#         - chunks: list of text chunks
#         - meta_out: list of metadata dicts for each chunk
#     """
#     chunks, meta_out = [], []
#     print('in the PREPROCESS')

#     # Iterate over each document (handles lists-of-lists)
#     for doc_id, doc in enumerate(itertools.chain.from_iterable(
#             d if isinstance(d, list) else [d] for d in docs)):

#         cleaned = clean(doc)  # Clean the text
#         source_meta = meta_in[doc_id]  # Original document metadata

#         # Chunk the cleaned text and generate metadata for each chunk
#         for ck_id, ck in enumerate(chunk(cleaned, max_tokens, overlap)):
#             chunk_meta = {
#                 "doc_id": doc_id,
#                 "chunk_id": ck_id,
#                 "tokens": len(ENCODER.encode(ck)),  # token count of the chunk
#                 **source_meta  # inherit and merge original metadata
#             }
#             chunks.append(ck)
#             meta_out.append(chunk_meta)

#     return chunks, meta_out