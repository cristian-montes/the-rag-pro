#!/usr/bin/env python
"""
Lightweight CLI for local Q&A that:
 • retrieves top-k chunks (BM25+FAISS),
 • feeds them to Mistral with an *answer-only-if-supported* prompt,
 • prints answer + structured citations.
"""

import os, re, textwrap, json
from dense.retrieval import retrieve
from load_mistral import load as load_llm
from dense.dense_corpus_loader.build_index import build

# Import sklearn stopwords for query cleaning
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Constants
K             = 4
MAX_GEN_TOK   = 512
STOP_TOKENS   = ["</s>", "###", "Answer:"]
DATA_DIR      = "data"
INDEX_DIR     = "index"

PROMPT_TMPL = """<|system|>
You are an expert assistant. Rely *only* on the provided context.
If the answer is not contained in it, reply exactly: "I don’t know.".
When you answer, append a line "Sources:" listing each cited chunk id.
<|user|>
Question: {question}

Context:
{context}
<|assistant|>
Answer:
"""

# Use sklearn's English stopwords as a set for quick lookup
SKLEARN_STOPWORDS = set(ENGLISH_STOP_WORDS)

def clean_query(query: str) -> str:
    # Remove punctuation and lowercase
    query = re.sub(r"[^\w\s]", " ", query.lower())
    query = re.sub(r"\s+", " ", query).strip()

    # Remove stopwords
    words = query.split()
    filtered = [w for w in words if w not in SKLEARN_STOPWORDS]
    return " ".join(filtered)

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

    faiss_path = os.path.join(INDEX_DIR, "dense_index.idx")

    if not os.path.exists(faiss_path):
        print("🔧 FAISS Index file not found. Building indexes...")
        build()  # This calls load_all_data() internally
        print("✅ All indexes built.\n")
    else:
        print("✅ Index files found. Skipping index building.\n")


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

        # Clean the query to remove stopwords like in preprocess
        cleaned_q = clean_query(q)
        if not cleaned_q:
            # If cleaning removed everything, fallback to original query
            cleaned_q = q

        hits = retrieve(cleaned_q, K)
        ctx = format_context(hits)
        prompt = PROMPT_TMPL.format(question=q, context=ctx)  # keep original q in prompt

        out = llm(
            prompt,
            max_tokens=MAX_GEN_TOK,
            temperature=0.2,
            top_p=0.8,
            stop=STOP_TOKENS
        )["choices"][0]["text"].strip()

        print("\n🧠", textwrap.fill(out, 100))

        cited_ids = set(map(int, re.findall(r"\[(\d+)\]", out)))

        if cited_ids:
            print("\n📚 Cited sources:")
            for idx, h in enumerate(hits, 1):
                if idx in cited_ids:
                    doc_snippet = h["doc"].strip().replace("\n", " ")
                    if len(doc_snippet) > 200:
                        doc_snippet = doc_snippet[:400].rstrip() + "..."
                    title = h["meta"].get("title", "?")
                    print(f"Title: {title} - \"{doc_snippet}\" ")
        else:
            print("\n📚 No specific sources cited.")

if __name__ == "__main__":
    main()
