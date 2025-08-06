#!/bin/bash

set -e
set -o pipefail

echo " Setting up only critical dependencies: pip, spaCy model, and Mistral weights..."

# 🛠 Ensure pip is available
echo "🧪 Ensuring pip is installed in the current Python environment..."
python -m ensurepip --upgrade

#  Install spaCy model only if not already downloaded
echo " Checking for spaCy language model: en_core_web_sm..."
if ! python -c "import spacy; spacy.load('en_core_web_sm')" &> /dev/null; then
  echo " Downloading spaCy model: en_core_web_sm..."
  python -m spacy download en_core_web_sm
else
  echo "spaCy model already installed."
fi

#  Ensure model directory exists
echo " Ensuring model directory exists..."
mkdir -p model

MODEL_NAME="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_PATH="model/$MODEL_NAME"

#  Download model if it doesn't already exist
if [ ! -f "$MODEL_PATH" ]; then
  echo " Downloading Mistral model: $MODEL_NAME"
  curl -L -o "$MODEL_PATH" "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/$MODEL_NAME"
else
  echo "Model already exists at $MODEL_PATH"
fi

echo "Minimal setup complete."

