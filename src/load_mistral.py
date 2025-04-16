from llama_cpp import Llama
import os

def load_mistral():
    model_path = "/Users/cristianmontes/Documents/dev/llama.cpp/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please make sure it's downloaded.")

    # Load the model with desired config
    model = Llama(
        model_path=model_path,
        n_ctx=4096,        # Max context tokens --> Longer contex for complex questions
        n_threads=8,       # Logical cores
        n_gpu_layers=35,   # Pushes most transformer layer to GPU via metal =  faster inference.
        use_mlock=True,    # Locks model into RAM - avoids slowdown from memory
        use_mmap=True,     # Speeds up loading model from disk.
        verbose=False,     # Keeps CLI clean. Can be set to "True" for debug.
    )

    return model