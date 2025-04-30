import os
import json
from sentence_transformers import SentenceTransformer

chunks_dir = "../data/chunks"

def get_chunks_data(chunks_dir):
    chunks_data = []
    for file_name in os.listdir(chunks_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(chunks_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, list):  # Ensure data is a list of chunks
                    for chunk in data:
                        if "file_name" not in chunk:
                            chunk["file_name"] = file_name.replace("_chunks.json", ".mp4")
                        chunks_data.extend(data)
                else:
                    print(f"Warning: File {file_name} does not contain a list.")
    return chunks_data

def create_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return embeddings

def main():
    chunks_data = get_chunks_data(chunks_dir)
    print(f"total chunks extracted: {len(chunks_data)}")

    if chunks_data:
        text_chunks = [chunk["text"] for chunk in chunks_data]
        # file_names = [chunk["file_name"] for chunk in chunks_data]
        embeddings = create_embeddings(text_chunks)
        print(f"embeddings shape: {embeddings.shape}")
    else:
        print("No data found to create embeddings.")

if __name__ == "__main__":
    main()
