#!/usr/bin/env python

from load_mistral import load_mistral
from retrieval import retrieve_text

def main():
    model, tokenizer = load_mistral()

    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        # Retrive context from corpus
        context = retrieve_text(query)
        print(f"\nRetrived context: {context}")

        # Pass context to the model
        inputs = tokenizer(query, return_tensors="pt").to("mps")
        outputs = model.generate(**inputs, max_new_tokens=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        #Display Result
        print(f"\nModel Response:\n{response}\n")

if __name__ == "__main__":
    main()