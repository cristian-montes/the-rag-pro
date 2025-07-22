"""
Preprocess module: clean, split large documents into manageable parts,
and perform sentence-based chunking using spaCy sentence segmentation.

This version splits large documents into smaller chunks before
feeding to spaCy to avoid exceeding spaCy's `nlp.max_length` limit,
which prevents memory issues and crashes on very long texts.
"""

import itertools
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load spaCy English model once globally for efficiency.
# spaCy provides robust sentence boundary detection and tokenization.
nlp = spacy.load("en_core_web_sm")

def clean(text: str) -> str:
    """
    Clean text by:
    - Lowercasing for normalization (so 'The' and 'the' are same).
    - Removing common English stopwords using sklearn's built-in list.
    
    Why?
    Stopwords add noise and don't help with retrieval relevance, so removing
    them improves precision and reduces index size.
    """
    words = text.lower().split()  # Split text into words after lowercasing
    filtered = [word for word in words if word not in ENGLISH_STOP_WORDS]  # Remove stopwords
    return ' '.join(filtered)  # Rejoin words into a clean string

def split_text(text: str, max_chunk_size=50000):
    """
    Split large texts into smaller chunks to avoid spaCy's max_length limit.
    
    spaCy raises errors if you feed it a text longer than ~1,000,000 chars,
    which also consumes huge memory. This function breaks text into chunks
    of max_chunk_size characters or less.
    
    It tries to split on natural boundaries like newlines or periods to avoid
    breaking sentences mid-way.
    
    Yields each chunk as a string.
    """
    if len(text) <= max_chunk_size:
        # Text is short enough, no splitting needed
        yield text
        return

    start = 0
    length = len(text)
    while start < length:
        # Tentatively set the end boundary for this chunk
        end = min(start + max_chunk_size, length)

        # Try to find a good split point on a newline within the current chunk
        breakpoint = text.rfind('\n', start, end)
        if breakpoint == -1 or breakpoint < start:
            # If no newline found, try splitting on a period '.'
            breakpoint = text.rfind('.', start, end)
        if breakpoint == -1 or breakpoint < start:
            # Fallback: just split at max_chunk_size boundary
            breakpoint = end
        else:
            # Include the delimiter character (newline or period)
            breakpoint += 1

        # Yield the chunk substring and move start forward
        yield text[start:breakpoint].strip()
        start = breakpoint

def chunk(text: str):
    """
    Main sentence-based chunking function.
    
    - Uses split_text() to break large text into manageable pieces.
    - For each smaller piece, runs spaCy's NLP pipeline to segment sentences.
    - Yields each sentence separately as an individual chunk.
    
    Why?
    Sentence-based chunks help retrieval by preserving semantic meaning and
    making chunks easier for the retriever and LLM to handle.
    """
    sentences = []
    for part in split_text(text):
        # Apply spaCy NLP pipeline to the smaller chunk
        doc = nlp(part)
        # Iterate over detected sentences (spaCy's sentence boundaries)
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if sent_text:
                sentences.append(sent_text)

    # Yield all collected sentences one by one
    for sentence in sentences:
        yield sentence

def preprocess(docs, meta_in):
    """
    Preprocessing main function.
    
    Inputs:
    - docs: list of raw document texts (can be nested lists).
    - meta_in: list of metadata dictionaries for each document (e.g., title, doc_id).
    
    Process:
    1. Flatten input docs list in case of nested lists.
    2. For each document:
       a. Clean text by removing stopwords and lowercasing.
       b. Sentence-chunk the cleaned text using the chunk() function.
       c. Attach metadata info to each sentence chunk.
    3. Return lists of sentence chunks and corresponding metadata.
    
    Why?
    Producing many short, semantically meaningful chunks with metadata
    improves retrieval precision and helps the LLM cite specific sources.
    """
    chunks, meta_out = [], []
    print('🧹 Starting preprocessing and sentence chunking...')

    # Flatten list of documents if nested, so each doc is processed individually
    for doc_id, doc in enumerate(itertools.chain.from_iterable(
            d if isinstance(d, list) else [d] for d in docs)):

        cleaned_text = clean(doc)  # Remove stopwords and lowercase

        source_meta = meta_in[doc_id]  # Retrieve original metadata for this doc

        # Sentence-based chunking: yield one sentence at a time
        for chunk_id, sentence in enumerate(chunk(cleaned_text)):
            chunk_meta = {
                "doc_id": doc_id,    # Index of document this chunk belongs to
                "chunk_id": chunk_id,  # Position of sentence chunk within the doc
                **source_meta         # Include original metadata fields
            }
            chunks.append(sentence)  # Append sentence chunk text
            meta_out.append(chunk_meta)  # Append corresponding metadata

    print(f"✅ Preprocessed {len(chunks)} sentence chunks across {len(docs)} documents.")
    return chunks, meta_out
