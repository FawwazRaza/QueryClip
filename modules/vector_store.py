import os
import json
import torch
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from transformers import AutoTokenizer, AutoModel

chukns_dir = "../data/chunks"
chroma_dir = "../data/db/chroma_db"
collection_name = "video_chunks"

def load_chunks(chunks_dir):
    chunks = []
    for file_name in os.listdir(chunks_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(chunks_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    chunks.extend(data)
    return chunks

def load_embeddings(embeddings_path):
    if embeddings_path.endswith(".pt"):
        return torch.load(embeddings_path)
    elif embeddings_path.endswith(".npy"):
        
        return np.load(embeddings_path)
    else:
        raise ValueError("Unsupported embeddings file format.")

def main():
    chunks = load_chunks(chukns_dir)
    print(f"Loaded {len(chunks)} chunks.")

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    texts = [chunk["text"] for chunk in chunks]
    # all_embeddings = []
    # for text in texts:
    #     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    #     with torch.no_grad():
    #         outputs = model(**inputs)
    #         embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
    #         all_embeddings.append(embedding)
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.extend(embeddings)
    print(f"Computed {len(all_embeddings)} embeddings.")

    client = chromadb.PersistentClient(path=chroma_dir)
    
    # if collection_name not in [c.name for c in client.list_collections()]:
    #     collection = client.create_collection(collection_name)
    # else:
    #     collection = client.get_collection(collection_name)
    if collection_name not in client.list_collections():
        collection = client.create_collection(collection_name)
    else:
        collection = client.get_collection(collection_name)

    metadatas = [
        {
            "start_time": chunk.get("start_time"),
            "end_time": chunk.get("end_time"),
            "text": chunk.get("text"),
            "file_name": chunk.get("file_name")
        }
        for chunk in chunks
    ]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        embeddings=all_embeddings,
        metadatas=metadatas,
        ids=ids,
        documents=texts  
    )
    print(f"Stored {len(all_embeddings)} embeddings in ChromaDB collection '{collection_name}'.")

if __name__ == "__main__":
    main()
