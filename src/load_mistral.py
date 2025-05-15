import os
from llama_cpp import Llama

MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", "/Users/cristianmontes/Documents/dev/llama.cpp/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
_llm_instance = None

def load_mistral():
    global _llm_instance
    if _llm_instance is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        print(f"Loading Mistral model from {MODEL_PATH}...")

        _llm_instance = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=35,
            use_mlock=True,
            use_mmap=True,
            chat_format="chatml",
            verbose=False,
        )

    return _llm_instance

