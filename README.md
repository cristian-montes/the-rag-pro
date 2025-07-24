# The RAG Pro

## Project Overview

**The RAG Pro** is a Retrieval-Augmented Generation (RAG) prototype that utilizes a local Large Language Model (LLM) to answer questions based on a custom indexed document corpus. It provides the avility to run either spare or dense retrival.



## Corpus Information

This project indexes and retrieves from specialized technical and historical documents, including:

1. [Archaeology, Anthropology, and Interstellar Communication (NASA)](https://www.nasa.gov/wp-content/uploads/2015/01/Archaeology_Anthropology_and_Interstellar_Communication_TAGGED.pdf?emrc=9e061d)  
2. [Governing the Moon (NASA SP-2024-4559)](https://www.nasa.gov/wp-content/uploads/2025/02/governing-the-moon-sp-2024-4559-ebook.pdf?emrc=686dfb5c7fb7d)  
3. [Rockets and People Volume 4 (NASA)](https://www.nasa.gov/wp-content/uploads/2015/04/621513main_rocketspeoplevolume4-ebook.pdf?emrc=4d370f)  

---

This implementation uses **Mistral 7B Instruct (v0.2.Q4\_K\_M.gguf)** model via `llama-cpp-python`. It is currently specialized in handling local document search and question answering, making it suitable for building personalized search or Q\&A systems.

> ⚠️ This setup is tailored for Apple Silicon (M1/M2/M3) Macs and is optimized for local environments using `uv` for dependency management.

---

## Features

* Local LLM with no external API calls
* Retrieval-augmented pipeline with both sparse (BM25) or dense (embedding-based) retrievers
* CLI for running queries against indexed documents

---

## Requirements

* MacOS with Apple M Series (M1/M2/M3) CPU
* Astral UV installed
* 8GB+ RAM recommended for model inference

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/the-rag-pro.git
cd the-rag-pro
```

### 2. Run the Setup Script

Make sure Conda is installed and accessible in your shell. Then, run:

```bash
chmod +x uv_setup.sh
./uv_setup.sh
```

This script will:

* Create and activate a Conda environment based on `environment.yml`
* Install necessary Python dependencies (e.g., `llama-cpp-python`, `chromadb`, `faiss-cpu`, `scikit-learn`, `spacy`, etc.)

> If environment creation fails, the script will stop and print an error so the user can investigate before retrying.

---

## File & Directory Structure

```
the-rag-pro/
├── model/                  # Directory to place your GGUF model file
│   └── mistral-7b-instruct-v0.2.Q4_K_M.gguf
├── src/
│   ├── load_model.py       # Handles model loading
│   ├── sparse_cli.py       # CLI script for querying documents
│   ├── dense_cli.py        # Alternative CLI using dense retriever
│   └── ...
├── environment.yml         # Conda environment file
├── uv_setup.sh                # Project setup script
└── README.md               # You are here
```

> 📂 Ensure your model is placed under `model/` directory and is correctly referenced in the script using full path or resolved using `os.path.abspath()`.

---

## Running the CLI

### Sparse Retriever Example

```bash
conda activate rag-pro
python src/sparse_cli.py --query "How do the Artemis Accords impact lunar governance?"
```

### Dense Retriever Example

```bash
conda activate rag-pro
python src/dense_cli.py --query "Summarize the challenges associated with resource extraction on the Moon as presented in the document"
```

### To Stop or Exit 

```bash
conda activate rag-pro
python src/dense_cli.py --query "exit"
```

> You can modify the indexed documents and embeddings inside the `src/` folder for your use case.

---

## Notes

* The Mistral model is loaded via `llama-cpp-python` and runs in `gguf` format optimized for CPU inference.
* File paths are resolved using `os.path.abspath(__file__)` to ensure portability.
* Troubleshooting tips for missing models or broken environments are printed to the console by the script.

---

