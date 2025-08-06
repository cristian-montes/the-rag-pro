"""
Local llama.cpp loader tuned for speed on Apple-silicon (M-series).
"""
import os
from llama_cpp import Llama

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PATH = os.getenv(
    "LLAMA_MODEL_PATH",
    os.path.join(BASE_DIR, "../model/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
)
_instance=None

def load(model_path=_DEFAULT_PATH, n_ctx=4096):
    global _instance
    if _instance is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        print(f"🔹 Loading model: {os.path.basename(model_path)}")
        _instance = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=40, 
            n_threads=os.cpu_count()//2 or 4,
            use_mmap=True,
            use_mlock=False,
            chat_format="chatml",
            verbose=False)
    return _instance
