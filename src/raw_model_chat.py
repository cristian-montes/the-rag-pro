from llama_cpp import Llama

# Path to your quantized Mistral GGUF model
MODEL_PATH = "/Users/cristianmontes/Documents/dev/llama.cpp/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Initialize the LLM
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,      # Adjust based on your CPU
    n_batch=128,
    verbose=True
)

# Chat loop
print("💬 Mistral Chat is ready! Type your question (or 'exit' to quit).")

while True:
    user_input = input("\n❓ You: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        print("👋 Exiting.")
        break

    prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{user_input}\n<|assistant|>\n"

    output = llm(
        prompt=prompt,
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
        stop=["</s>", "<|user|>"]
    )

    answer = output["choices"][0]["text"].strip()
    print(f"🤖 Mistral: {answer}")
