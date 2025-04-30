import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel
import torch

chroma_dir = "../data/db/chroma_db"
collect_name = "video_chunks"
modeln = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(modeln)
model = AutoModel.from_pretrained(modeln)

def embed_text(text):
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
    return embedding

class ChromaRetriever:
    def __init__(self, chroma_db_dir=chroma_dir, collection_name=collect_name, default_k=5):
        self.client = chromadb.PersistentClient(path=chroma_db_dir)
        self.collection = self.client.get_collection(collection_name)
        self.default_k = default_k

    def similarity_search(self, query, k=None):
        if k is None:
            k = self.default_k
        query_embedding = embed_text(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['metadatas', 'documents', 'distances']
        )
        hits = []
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

retriever_chain = ChromaRetriever().as_retriever(search_kwargs={"k": 5})

if __name__ == "__main__":
    query = "Please provide mr information about Fawwaz Raza."
    results = retriever_chain(query, k=5)
    for i, chunk in enumerate(results, 1):
        print(f"Chunk {i}:")
        print(f"File Name: {chunk['file_name']}")
        print(f"Text: {chunk['text']}")
        print(f"Start: {chunk['start_time']}s, End: {chunk['end_time']}s, Similarity: {chunk['similarity']:.3f}\n")
