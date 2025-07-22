#!/bin/bash

set -e  # Exit on error
set -o pipefail

# ----- Config -----
MODEL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MODEL_DIR="./local_models"
MODEL_FILE="mistral-7b-instruct-v0.1.Q4_K_M.gguf"
ENV_NAME="the_rag_pro"

# ----- Check: Conda -----
if ! command -v conda &> /dev/null; then
  echo "❌ Conda is not installed. Please install Miniconda or Anaconda first."
  exit 1
fi

# ----- Step 1: Create Conda Environment -----
echo "🔧 Creating Conda environment '$ENV_NAME'..."
conda env create -n $ENV_NAME -f environment_shared.yml || {
  echo "✅ Environment likely already exists. Skipping."
}

# ----- Step 2: Download Mistral GGUF Model -----
echo "📥 Downloading Mistral model..."
mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
  echo "✅ Model already exists at $MODEL_DIR/$MODEL_FILE"
else
  curl -L -o "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL"
  echo "✅ Model downloaded to $MODEL_DIR/$MODEL_FILE"
fi

# ----- Step 3: Final Info -----
echo ""
echo "✅ Setup complete!"
echo ""
echo "👉 To activate your environment and run the CLI:"
echo ""
echo "   conda activate $ENV_NAME"
echo "   python src/cli.py"
echo ""
echo "🧠 Make sure your CLI script is pointed to use the local model at $MODEL_DIR/$MODEL_FILE"
