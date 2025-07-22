#!/bin/bash

echo "🔧 Starting setup for the-rag-pro on MacOS (M-series)..."

# Step 1: Confirm Conda is available
if ! command -v conda &> /dev/null
then
    echo "❌ Conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Step 2: Create conda environment
echo "📦 Creating conda environment from environment.yml..."
conda env create -f environment.yml --force

# Step 3: Activate environment
echo "✅ Environment created. You can activate it using:"
echo ""
echo "   conda activate rag-pro"
echo ""

# Step 4: Create models directory if needed
MODEL_DIR="models"
mkdir -p "$MODEL_DIR"

# Step 5: Download Mistral model if not already there
MODEL_NAME="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/$MODEL_NAME"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

if [ -f "$MODEL_PATH" ]; then
    echo "✅ Model already downloaded: $MODEL_PATH"
else
    echo "⬇️  Downloading Mistral model..."
    curl -L -o "$MODEL_PATH" "$MODEL_URL"

    if [ $? -ne 0 ]; then
        echo "❌ Failed to download model. Please check the URL or your internet connection."
        exit 1
    else
        echo "✅ Model downloaded to $MODEL_PATH"
    fi
fi

echo "🎉 Setup complete! You can now activate the environment and run your project:"
echo ""
echo "   conda activate rag-pro"
echo "   python src/dense/cli.py --help"

echo "🧠 Make sure your CLI script is pointed to use the local model at $MODEL_DIR/$MODEL_FILE"
