"""ChromaDB vector store with sentence-transformers embeddings."""

import hashlib
from pathlib import Path
from typing import List, Optional

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_COLLECTION_NAME = "youtube_transcripts"
_DB_PATH = Path(__file__).parent.parent / "data" / "chroma_db"

_encoder: Optional[SentenceTransformer] = None


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(_EMBEDDING_MODEL)
    return _encoder


def _embed(texts: List[str]) -> List[List[float]]:
    embeddings = _get_encoder().encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
    )
    return embeddings.tolist()


def _get_client() -> chromadb.PersistentClient:
    _DB_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(_DB_PATH))


def _get_collection(client: chromadb.PersistentClient):
    try:
        return client.get_collection(_COLLECTION_NAME)
    except Exception:
        return client.create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


def add_documents(documents: List[Document]) -> int:
    """Store LangChain Documents in ChromaDB.

    Uses content-based IDs so re-processing the same video is idempotent.

    Args:
        documents: Chunked Document objects with metadata.

    Returns:
        Number of chunks stored.
    """
    if not documents:
        return 0

    client = _get_client()
    collection = _get_collection(client)

    texts = [doc.page_content for doc in documents]
    metadatas = [
        {k: str(v) if v is not None else "" for k, v in doc.metadata.items()}
        for doc in documents
    ]
    ids = [
        hashlib.md5(
            f"{doc.metadata.get('video_id', '')}_{doc.metadata.get('chunk_index', i)}".encode()
        ).hexdigest()
        for i, doc in enumerate(documents)
    ]
    embeddings = _embed(texts)

    collection.upsert(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids,
    )

    return len(documents)


def query_documents(query: str, top_k: int = 4) -> List[dict]:
    """Retrieve the most relevant document chunks for a query.

    Args:
        query: User query string.
        top_k: Maximum number of results to return.

    Returns:
        List of dicts with keys: text, metadata, similarity.
    """
    client = _get_client()
    collection = _get_collection(client)

    count = collection.count()
    if count == 0:
        return []

    actual_k = min(top_k, count)
    query_embedding = _embed([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=actual_k,
        include=["documents", "metadatas", "distances"],
    )

    hits: List[dict] = []
    if results and results.get("ids") and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            hits.append(
                {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": round(1.0 - results["distances"][0][i], 4),
                }
            )

    return hits


def list_indexed_videos() -> List[dict]:
    """Return unique videos currently stored in ChromaDB.

    Returns:
        List of dicts with video_id, title, url, playlist_name.
    """
    client = _get_client()
    collection = _get_collection(client)

    if collection.count() == 0:
        return []

    all_items = collection.get(include=["metadatas"])
    seen: dict = {}

    for meta in all_items.get("metadatas") or []:
        vid_id = meta.get("video_id", "")
        if vid_id and vid_id not in seen:
            seen[vid_id] = {
                "video_id": vid_id,
                "title": meta.get("title", "Unknown"),
                "url": meta.get("url", ""),
                "playlist_name": meta.get("playlist_name", ""),
            }

    return list(seen.values())


def clear_collection() -> None:
    """Delete all data from the ChromaDB collection."""
    client = _get_client()
    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass
