import os
import pandas as pd
import json

def load_csv(data_path):
    csv_texts = []
    metadata = []

    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            file_path = os.path.join(data_path, file)
            try:
                df = pd.read_csv(file_path)
                # Extract text per cell with location info
                for row_idx, row in df.iterrows():
                    for col_idx, value in enumerate(row):
                        text = str(value)
                        if text.strip():
                            csv_texts.append(text)
                            metadata.append({
                                "source": "CSV",
                                "filename": file,
                                "filepath": file_path,
                                "row": row_idx,
                                "column": df.columns[col_idx]
                            })

            except Exception as e:
                print(f"Failed to load {file}: {e}")
    return csv_texts, metadata

if __name__ == "__main__":
    csv_dir = 'data/csvs'
    csv_texts, metadata = load_csv(csv_dir)
    with open("data/csvs/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Loaded {len(csv_texts)} CSV cells with metadata.")
