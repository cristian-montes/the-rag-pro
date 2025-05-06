import os
import pandas as pd
import json

def load_csv(data_path):
    csv_texts = []
    metadata = []  # List to store metadata

    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            file_path = os.path.join(data_path, file)
            try:
                df = pd.read_csv(file_path)
                # Combine all text data from the CSV into a single string
                csv_text = "\n".join(df.astype(str).stack())
                csv_texts.append(csv_text)

                # Create metadata for this file
                metadata.append({
                    "source": "CSV",
                    "filename": file,
                    "url": file_path,
                    "rows": len(df),
                    "columns": len(df.columns)
                })

            except Exception as e:
                print(f"Failed to load {file}: {e}")
    return csv_texts, metadata

if __name__ == "__main__":
    csv_dir = 'data/csvs'
    csv_texts, metadata = load_csv(csv_dir)

    # Save metadata to a json file
    with open("data/csvs/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Loaded {len(csv_texts)} CSV files with metadata.")





# corpus_loader/load_csv.py
# import os
# import pandas as pd

# def load_csv(data_path):
#     csv_texts = []
#     for file in os.listdir(data_path):
#         if file.endswith(".csv"):
#             file_path = os.path.join(data_path, file)
#             try:
#                 df = pd.read_csv(file_path)
#                 # Combine all text data from the CSV into a single string
#                 csv_texts.append("\n".join(df.astype(str).stack()))
#             except Exception as e:
#                 print(f"Failed to load {file}: {e}")
#     return csv_texts

# if __name__ == "__main__":
#     csv_dir = 'data/csvs'
#     csv_texts = load_csv(csv_dir)
#     print(f"Loaded {len(csv_texts)} CSV files.")
