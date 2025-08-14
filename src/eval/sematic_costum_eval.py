# eval_custom.py
import sys
import os
import json
from sentence_transformers import SentenceTransformer, util
from sparse_cli import ensure_ready, load_llm, answer_question


# Add parent folder ("src") to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(EMBED_MODEL)

def load_eval_data():
    path = os.path.join(os.path.dirname(__file__), "eval_golden_qa.json")
    with open(path, "r") as f:
        return json.load(f) 

def semantic_similarity(a, b):
    emb_a = model.encode(a, convert_to_tensor=True)
    emb_b = model.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb_a, emb_b).item()

def evaluate():
    ensure_ready()
    llm = load_llm()
    qa_pairs = load_eval_data()
    results = []

    for qa in qa_pairs:
        question = qa["question"]
        golden_answer = qa["golden_answer"]

        result = answer_question(question, llm=llm)
        retrieved_answer = result["answer"]

        score = semantic_similarity(golden_answer, retrieved_answer)

        results.append({
            "question": question,
            "golden_answer": golden_answer,
            "retrieved_answer": retrieved_answer,
            "similarity_score": score
        })

    # Save results
    with open("eval/results/custom_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    avg_score = sum(r["similarity_score"] for r in results) / len(results)
    print(f"Average similarity score: {avg_score:.3f}")

if __name__ == "__main__":
    evaluate()
