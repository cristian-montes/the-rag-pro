"""
One-shot pipeline: load raw sources → preprocess → (re)build indexes.
Run whenever you add new docs.
"""
from corpus_loader.load_all_data import load_all_data
from src.corpus_loader.preprocess import preprocess
from src.corpus_loader.build_index_bm25 import build        # re-exports from previous script

def main():
    print("🗄️  Building indexes from scratch …")
    build()
    print("✅ Done.")

if __name__ == "__main__":
    main()
