#!/bin/bash

set -e
set -o pipefail

echo "🔧 Starting setup for the-rag-pro on macOS (M-series) with uv and pyproject.toml..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
  echo " 'uv' command not found. Please install uv first: https://astral.sh/uv"
  exit 1
fi

# Remove existing virtual environment if any
if [ -d ".venv" ]; then
  echo " Removing existing .venv directory..."
  rm -rf .venv
fi

echo " Creating and syncing virtual environment using pyproject.toml..."
uv sync

echo "Activating the virtual environment..."
source .venv/bin/activate

# 🛠 Ensure pip is installed (some uv-created envs may not include it)
echo "Ensuring pip is installed in the virtual environment..."
python -m ensurepip --upgrade

echo " Installing spaCy language model en_core_web_sm..."
python -m spacy download en_core_web_sm

echo "Setting up model directory..."
mkdir -p model

MODEL_NAME="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_PATH="model/$MODEL_NAME"

if [ ! -f "$MODEL_PATH" ]; then
  echo "⬇  Downloading Mistral model: $MODEL_NAME"
  curl -L -o "$MODEL_PATH" "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/$MODEL_NAME"
else
  echo " Model already exists at $MODEL_PATH"
fi

echo " Setup complete. Activate the environment with:"
echo ""
echo "    source .venv/bin/activate"
echo ""
