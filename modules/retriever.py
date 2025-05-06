import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel
import torch
import os
import numpy as np
import hashlib

chroma_dir = "../data/db/chroma_db"
os.makedirs(chroma_dir, exist_ok=True)
collect_name = "video_chunks"
modeln = "sentence-transformers/all-MiniLM-L6-v2"

try:
    tokenizer = AutoTokenizer.from_pretrained(modeln)
    model = AutoModel.from_pretrained(modeln)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Using fallback embedding method")
    tokenizer = None
    model = None

def embed_text(text):
    if tokenizer is not None and model is not None:
        try:
            inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            return embedding
        except Exception as e:
            print(f"Error in embedding: {str(e)}")
            # Fall back to simple hash-based embedding
            return simple_embedding(text)
    else:
        return simple_embedding(text)

def simple_embedding(text):
    
    
    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()
    
    embedding = np.array([float(b) for b in hash_bytes]) / 255.0
    
    target_dim = 384
    if len(embedding) < target_dim:
        embedding = np.pad(embedding, (0, target_dim - len(embedding)))
    else:
        embedding = embedding[:target_dim]
        
    return embedding


class ChromaRetriever:
    def __init__(self, chroma_db_dir=chroma_dir, collection_name=collect_name, default_k=5):
        try:
            self.client = chromadb.PersistentClient(path=chroma_db_dir)
            try:
                self.collection = self.client.get_collection(collection_name)
                print(f"Successfully connected to collection: {collection_name}")
            except Exception as e:
                print(f"Collection {collection_name} not found, creating a new one")
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "Video chunks collection"}
                )
        except Exception as e:
            print(f"Error initializing ChromaDB: {str(e)}")
            # Create a dummy fallback for testing purposes
            self.client = None
            self.collection = None
        
        self.default_k = default_k

    def similarity_search(self, query, k=None):
        if k is None:
            k = self.default_k
            
        if self.collection is None:
            print("Warning: ChromaDB collection not available. Returning empty results.")
            return []
            
        try:
            query_embedding = embed_text(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=['metadatas', 'documents', 'distances']
            )
            
            hits = []
            if results and 'ids' in results and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    meta = results['metadatas'][0][i]
                    text = results['documents'][0][i]
                    distance = results['distances'][0][i]
                    hits.append({
                        "text": text,
                        "start_time": meta.get("start_time"),
                        "end_time": meta.get("end_time"),
                        "file_name": meta.get("file_name"),
                        "similarity": 1 - distance  
                    })
            return hits
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            return []

    def as_retriever(self, search_type="similarity", search_kwargs=None):
        def retriever(query, k=None):
            if k is not None:
                num_results = k
            elif search_kwargs and "k" in search_kwargs:
                num_results = search_kwargs["k"]
            else:
                num_results = self.default_k
            results = self.similarity_search(query, k=num_results)
            return results 
        return retriever

# Initialize the retriever chain
retriever_chain = ChromaRetriever().as_retriever(search_kwargs={"k": 5})

if __name__ == "__main__":
    query = "Please provide information about video content."
    results = retriever_chain(query, k=5)
    for i, chunk in enumerate(results, 1):
        print(f"Chunk {i}:")
        print(f"File Name: {chunk['file_name']}")
        print(f"Text: {chunk['text']}")
        print(f"Start: {chunk['start_time']}s, End: {chunk['end_time']}s, Similarity: {chunk['similarity']:.3f}\n")