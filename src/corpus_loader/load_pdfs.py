import os
import fitz  # PyMuPDF
from .download_pdfs import main as download_pdf_main

def load_pdfs(path):
    download_pdf_main()
    texts = []
    metadata = []

    for file in os.listdir(path):
        if file.endswith(".pdf"):
            file_path = os.path.join(path, file)
            try:
                with fitz.open(file_path) as doc:
                    text = ""
                    for page in doc:
                        text += page.get_text()

                    texts.append(text)
                    metadata.append({
                        "source": "pdf",
                        "filename": file,
                        "filepath": file_path
                    })
            except Exception as e:
                print(f"Failed to load {file}: {e}")

    return texts, metadata

if __name__ == "__main__":
    pdf_dir = 'data/pdfs'
    pdf_texts, pdf_meta = load_pdfs(pdf_dir)
    print(f"Loaded {len(pdf_texts)} PDF files.")
    print("Sample metadata:", pdf_meta[0] if pdf_meta else "None")







# import os
# import fitz

# def load_pdfs(path):
#      corpus = []
#      for file in os.listdir(path):
#         if file.endswith(".pdf"):
#             file_path = os.path.join(path, file)
#             try:
#                 with fitz.open(file_path) as doc:
#                     text = ""
#                     for page in doc:
#                         text += page.get_text()
#                     corpus.append(text)
#             except Exception as e:
#                 print(f"Failed to load {file}: {e}")
#      return corpus

# if __name__ == "__main__":
#     pdf_dir = 'data/pdfs'
#     pdf_texts = load_pdfs(pdf_dir)
#     print(f"Loaded {len(pdf_texts)} PDF files.")
