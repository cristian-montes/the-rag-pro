"""
Text cleaning + chunking that keeps accurate LLaMA token counts and
emits rich, per-chunk metadata for citations.
Supports SLIDING WINDOW strategies for both minimal duplication or
maximum semantic coherence.
"""
import html
import unicodedata
import re, itertools, tiktoken
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ── Token encoder ──
# Approximates LLaMA's tokenization for accurate token length limits
ENCODER = tiktoken.get_encoding("cl100k_base")  # similar to LLaMA 7B tokenizer


def clean(text: str) -> str:
    # Decode HTML entities like &nbsp;, &#x2019;, etc.
    text = html.unescape(text)
    
    # Normalize Unicode (e.g., \u2019 → ’)
    text = unicodedata.normalize("NFKC", text)

    # Remove soft hyphens and control characters
    text = re.sub(r'[\u00ad\u200b\u200e\u200f]', '', text)

    # Replace smart quotes/dashes with ASCII equivalents
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")

    # Standardize line breaks and spacing
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)         # Collapse 3+ line breaks to 2
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)   # Single newline → space

    # Keep words, punctuation, numbers; strip odd symbols
    text = re.sub(r"[^\w\s\.,:;?!'-]", '', text)

    # Lowercase, remove stop words
    words = text.lower().split()

    return ' '.join(words).strip()


def chunk(text: str, max_tokens=400, overlap=30, strategy="semantic"):
    """
    Chunk text with a sliding window.

    strategy: "semantic" -> high overlap (better coherence)
              "minimal"  -> low overlap (faster, cheaper)
    """
    words = text.split()
    ptr = 0

    # Adjust overlap based on strategy
    if strategy == "minimal":
        overlap = 0
    elif strategy == "semantic":
        overlap = overlap
    else:
        raise ValueError("Invalid strategy. Choose 'semantic' or 'minimal'.")

    while ptr < len(words):
        chunk_words = words[ptr:ptr + max_tokens]

        # Ensure token count is within limit
        while len(ENCODER.encode(" ".join(chunk_words))) > max_tokens:
            chunk_words = chunk_words[:-1]

        yield " ".join(chunk_words)

        # Slide window forward
        ptr += max_tokens - overlap

# ── Main preprocessing function ──
# Cleans and chunks each document; returns text chunks and detailed metadata
def preprocess(docs, meta_in, *, max_tokens=400, overlap=30, strategy="semantic"):
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
        for ck_id, ck in enumerate(chunk(cleaned, max_tokens, overlap, strategy)):
            chunk_meta = {
                "doc_id": doc_id,
                "chunk_id": ck_id,
                "tokens": len(ENCODER.encode(ck)),  # token count of the chunk
                **source_meta  # inherit and merge original metadata
            }
            chunks.append(ck)
            meta_out.append(chunk_meta)

    return chunks, meta_out






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
# def chunk(text: str, max_tokens=400, overlap=50):
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