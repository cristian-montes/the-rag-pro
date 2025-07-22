#!/bin/bash

set -e  # Exit on first error
set -o pipefail

echo "🔧 Starting setup for the-rag-pro on macOS (M-series)..."

# Get env name from environment.yml
ENV_NAME=$(grep '^name:' environment.yml | awk '{print $2}')

if [ -z "$ENV_NAME" ]; then
  echo "❌ Could not determine environment name from environment.yml"
  exit 1
fi

echo "📦 Checking for existing conda environment: $ENV_NAME..."
if conda info --envs | grep -q "^$ENV_NAME[[:space:]]"; then
  echo "🧹 Removing existing environment '$ENV_NAME'..."
  conda env remove -n "$ENV_NAME"
fi

echo "📦 Creating conda environment from environment.yml..."
if ! conda env create --file environment.yml; then
  echo "❌ Failed to create the conda environment '$ENV_NAME'."
  echo "💡 Please check that your environment.yml is valid and try again."
  exit 1
fi

echo "✅ Conda environment '$ENV_NAME' created successfully."

echo "📥 Installing spaCy model en_core_web_sm..."
conda run -n "$ENV_NAME" python -m spacy download en_core_web_sm

echo "📁 Setting up model directory..."
mkdir -p model

# If the model doesn't exist, download it
MODEL_NAME="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_PATH="model/$MODEL_NAME"

if [ ! -f "$MODEL_PATH" ]; then
  echo "⬇️  Downloading Mistral model: $MODEL_NAME"
  curl -L -o "$MODEL_PATH" https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/$MODEL_NAME
else
  echo "✅ Model already exists at $MODEL_PATH"
fi

echo "✅ Setup complete. Activate the environment with:"
echo ""
echo "    conda activate $ENV_NAME"
echo ""
