# corpus_loader/load_csv.py
import os
import pandas as pd

def load_csv(data_path):
    csv_texts = []
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            file_path = os.path.join(data_path, file)
            try:
                df = pd.read_csv(file_path)
                # Combine all text data from the CSV into a single string
                csv_texts.append("\n".join(df.astype(str).stack()))
            except Exception as e:
                print(f"Failed to load {file}: {e}")
    return csv_texts

if __name__ == "__main__":
    csv_dir = 'data/csvs'
    csv_texts = load_csv(csv_dir)
    print(f"Loaded {len(csv_texts)} CSV files.")
