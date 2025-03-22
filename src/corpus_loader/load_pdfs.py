import os
import fitz

def load_pdfs(path='./data/nasa_reports'):
     corpus = []
     for file in os.listdir(path):
        if file.endswith(".pdf"):
            file_path = os.path.join(path, file)
            try:
                with fitz.open(file_path) as doc:
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    corpus.append(text)
            except Exception as e:
                print(f"Failed to load {file}: {e}")
     return corpus

if __name__ == "__main__":
    pdf_dir = 'data/pdfs'
    pdf_texts = load_pdfs(pdf_dir)
    print(f"Loaded {len(pdf_texts)} PDF files.")