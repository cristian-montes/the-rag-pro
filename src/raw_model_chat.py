from llama_cpp import Llama

MODEL_PATH = "/Users/cristianmontes/Documents/dev/llama.cpp/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=8192,
    n_threads=8,
    n_gpu_layers=-1,
    n_batch=512,
    verbose=True
)

# Chat loop
print("💬 Mistral Chat is ready! Type your question, press enter, then paste your long text.")
print("Type 'END' on a new line to finish your input (or 'exit' to quit the chat).")

while True:
    user_input_lines = []
    print("\n❓ You:")
    while True:
        line = input()
        if line.lower() == "end":
            break
        user_input_lines.append(line)
    
    user_input = "\n".join(user_input_lines)
    
    if user_input.lower().strip() in {"exit", "quit"}:
        print("👋 Exiting.")
        break
        
    prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{user_input}\n<|assistant|>\n"

    output = llm(
        prompt=prompt,
        max_tokens=2048,
        temperature=0.7,
        top_p=0.9,
        stop=["</s>", "<|user|>"]
    )

    answer = output["choices"][0]["text"].strip()
    print(f"🤖 Mistral: {answer}")