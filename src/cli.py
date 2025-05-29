#!/usr/bin/env python
"""
Lightweight CLI for local Q&A that:
 • retrieves top-k chunks (BM25+FAISS),
 • feeds them to Mistral with an *answer-only-if-supported* prompt,
 • prints answer + structured citations.
"""

import os, re, textwrap, json
from retrieval import retrieve
from load_mistral import load as load_llm
from corpus_loader.build_index import build


# Constants
K             = 6
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
    faiss_path = os.path.join(INDEX_DIR, "faiss.idx")

    if not os.path.exists(bm25_path) or not os.path.exists(faiss_path):
        print("🔧 Index files not found. Building indexes...")
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

        hits = retrieve(q, K)
        ctx = format_context(hits)
        prompt = PROMPT_TMPL.format(question=q, context=ctx)

        out = llm(
            prompt,
            max_tokens=MAX_GEN_TOK,
            temperature=0.2,
            top_p=0.9,
            stop=STOP_TOKENS
        )["choices"][0]["text"].strip()

        print("\n🧠", textwrap.fill(out, 100))

        # Extract cited source numbers from the answer
        cited_ids = set(map(int, re.findall(r"\[(\d+)\]", out)))

        if cited_ids:
            print("\n📚 Cited sources:")
            for idx, h in enumerate(hits, 1):
                if idx in cited_ids:
                    doc_snippet = h["doc"].strip().replace("\n", " ")
                    # metaatos = h["meta"]
                    # print(metaatos)

                    if len(doc_snippet) > 200:
                        doc_snippet = doc_snippet[:400].rstrip() + "..."
                    # doc_id = h["meta"].get("doc_id", "?")
                    # chunk_id = h["meta"].get("chunk_id", "?")
                    title = h["meta"].get("title", "?")
                    # print(f"[{idx}] \"{doc_snippet}\" (doc_id: {doc_id}, chunk_id: {chunk_id}), title: {title}")
                    print(f"Title: {title} - \"{doc_snippet}\" ")
        else:
            print("\n📚 No specific sources cited.")

if __name__ == "__main__":
    main()



