import os

def retrieve_text(query):
    corpus_path = "data/corpus"
    results = []

    for file in os.listdir(corpus_path):
        with open(os.path.join(corpus_path, file), 'r') as f:
            text = f.read()
            if query.lower() in text.lower():
                results.append(text)

    return results

if __name__ == "__main__":
    query = "quantization"
    results = retrieve_text(query)
    print(f"\nResults for query '{query}':")
    for res in results:
        print(res)