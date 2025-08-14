#!/usr/bin/env python
"""
Lightweight CLI for local Q&A that:
 • retrieves top-k chunks (BM25),
 • feeds them to Mistral with an *answer-only-if-supported* prompt,
 • prints answer + structured citations.
"""

import os
import re
import textwrap
import json
from sparse.retrieval_bm25 import retrieve
from load_mistral import load as load_llm
from sparse.sparse_corpus_loader.build_index_bm25 import build
import spacy

# Constants
K             = 5  # max chunks to avoid too many tokens
MAX_GEN_TOK   = 512
STOP_TOKENS   = ["</s>", "###", "Answer:"]
DATA_DIR      = "data"
INDEX_DIR     = "index"

PROMPT_TMPL = """<|system|>
You are an expert assistant for a retrieval-augmented generation system.
Follow these rules strictly:
1. Only use facts from the provided CONTEXT.
2. If the CONTEXT does not contain the answer, reply exactly: "I don’t know."
3. Every answer MUST include a "Sources:" line listing ONLY the chunk IDs that contain facts you directly used.
4. Do NOT list all chunk IDs. If none of the chunks are relevant, reply exactly: "I don’t know."
<|user|>
Question: {question}

Context:
{context}
<|assistant|>
Answer:
"""



# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

def clean_query(query: str) -> str:
    query = query.lower()
    query = re.sub(r"[^\w\s]", ' ', query)
    query = re.sub(r"\s+", " ", query).strip()
    doc = nlp(query)
    lemmas = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(lemmas)

def format_context(hits):
    lines = []
    for i, h in enumerate(hits, 1):
        meta = json.dumps(h["meta"], ensure_ascii=False)
        lines.append(f"[{i}] {h['doc']}\nMETA: {meta}\n")
    return "\n".join(lines)

def ensure_ready():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

    print("\n🔍 Checking for existing data and indexes...")

    bm25_path = os.path.join(INDEX_DIR, "bm25.pkl")
  
    if not os.path.exists(bm25_path):
        print("🔧 Index files not found. Building indexes...")
        build()  # This calls load_all_data() internally
        print("✅ All indexes built.\n")
    else:
        print("✅ Index files found. Skipping index building.\n")

# === New function to answer a single question ===
def answer_question(question: str, k: int = K, llm=None) -> dict:
    """
    Given a question string, retrieve top-k chunks and generate answer with citations.
    Returns dict with keys:
      - 'answer': str, generated answer text
      - 'cited_chunks': list of dicts, each dict with keys: idx (int), title (str), snippet (str)
    """
    if llm is None:
        raise ValueError("Language model instance 'llm' must be provided.")

    cleaned_q = clean_query(question)
    if not cleaned_q:
        cleaned_q = question

    hits = retrieve(cleaned_q, k)
    ctx = format_context(hits)
    prompt = PROMPT_TMPL.format(question=question, context=ctx)

    out = llm(
        prompt,
        max_tokens=MAX_GEN_TOK,
        temperature=0.2,
        top_p=0.8,
        stop=STOP_TOKENS
    )["choices"][0]["text"].strip()

    #Guard Rail - to preven model from pulling from memory.
    has_sources = "Sources:" in out and re.search(r"\[\d+\]", out)

    cited_chunks = []
    if has_sources:
      # Extract cited IDs
        cited_ids = set(map(int, re.findall(r"\[(\d+)\]", out)))
        valid_ids = {i for i in cited_ids if 1 <= i <= len(hits)}

        # Check relevance: only keep chunks whose text overlaps with the answer
        relevant_ids = set()
        for idx in valid_ids:
            chunk_text = hits[idx-1]["doc"].lower()
            if any(word in chunk_text for word in out.lower().split()):
                relevant_ids.add(idx)

        # If no relevant sources remain, wipe them out
        if not relevant_ids:
            out = "I don’t know."
            cited_chunks = []
        else:
            cited_chunks = [
                {"idx": idx, "title": hits[idx-1]["meta"].get("title", "?"),
                "snippet": hits[idx-1]["doc"][:100] + "..."}
                for idx in relevant_ids
            ]


    return {
        "answer": out,
        "cited_chunks": cited_chunks,
    }

def main():
    ensure_ready()
    llm = load_llm()
    print("🔸 Ask anything (type 'exit' to quit).")
    while True:
        q = input("\n❓ ").strip()
        if q.lower() == "exit":
            print("👋 Goodbye!")
            break
        if not re.search(r"\w", q):
            continue

        result = answer_question(q, llm=llm)

        print("\n🧠", textwrap.fill(result["answer"], 100))

        if result["cited_chunks"]:
            print("\n📚 Cited sources:")
            for c in result["cited_chunks"]:
                print(f"Title: {c['title']} - \"{c['snippet']}\"")
        else:
            print("\n📚 No specific sources cited.")

if __name__ == "__main__":
    main()
