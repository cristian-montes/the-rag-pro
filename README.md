# The RAG Pro

## Project Overview

**The RAG Pro** is a Retrieval-Augmented Generation (RAG) prototype that utilizes a local Large Language Model (LLM) to answer questions based on a custom indexed document corpus. It combines both sparse and dense retrieval techniques for optimal performance.

This implementation uses **Mistral 7B Instruct (v0.2.Q4\_K\_M.gguf)** model via `llama-cpp-python`. It is currently specialized in handling local document search and question answering, making it suitable for building personalized search or Q\&A systems.

> ⚠️ This project is designed specifically for **MacOS M Series (Apple Silicon)** machines. You must have **Conda or Miniconda** installed prior to running the setup script.

---

## Features

* Local LLM with no external API calls
* Retrieval-augmented pipeline with both sparse (BM25) and dense (embedding-based) retrievers
* CLI for running queries against indexed documents

---

## Requirements

* MacOS with Apple M Series (M1/M2/M3) CPU
* Conda or Miniconda installed
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
chmod +x setup.sh
./setup.sh
```

This script will:

* Create and activate a Conda environment based on `environment.yml`
* Install necessary Python dependencies (e.g., `llama-cpp-python`, `chromadb`, `faiss-cpu`, `scikit-learn`, `spacy`, etc.)
* Download any required NLP models (if applicable)

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
├── setup.sh                # Project setup script
└── README.md               # You are here
```

> 📂 Ensure your model is placed under `model/` directory and is correctly referenced in the script using full path or resolved using `os.path.abspath()`.

---

## Running the CLI

### Sparse Retriever Example

```bash
conda activate rag-pro
python src/sparse_cli.py --query "What is vector search?"
```

### Dense Retriever Example

```bash
conda activate rag-pro
python src/dense_cli.py --query "Tell me about retrieval models."
```

> You can modify the indexed documents and embeddings inside the `src/` folder for your use case.

---

## Notes

* The Mistral model is loaded via `llama-cpp-python` and runs in `gguf` format optimized for CPU inference.
* File paths are resolved using `os.path.abspath(__file__)` to ensure portability.
* Troubleshooting tips for missing models or broken environments are printed to the console by the script.

---

