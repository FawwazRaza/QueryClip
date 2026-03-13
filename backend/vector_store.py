import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_COLLECTION_NAME = "youtube_transcripts"
_DB_PATH = Path(__file__).parent.parent / "data" / "chroma_db"
_UPSERT_BATCH = 200

_encoder: Optional[SentenceTransformer] = None

if not os.getenv("HF_TOKEN"):
    logger.info(
        "HF_TOKEN not set — using anonymous HuggingFace Hub access. "
        "Set HF_TOKEN in .env for higher rate limits."
    )


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(_EMBEDDING_MODEL)
    return _encoder


def _embed(texts: List[str]) -> List[List[float]]:
    embeddings = _get_encoder().encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
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

    for start in range(0, len(documents), _UPSERT_BATCH):
        end = start + _UPSERT_BATCH
        collection.upsert(
            embeddings=embeddings[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )

    return len(documents)


def query_documents(
    query: str,
    top_k: int = 4,
    video_id: Optional[str] = None,
) -> List[dict]:
    client = _get_client()
    collection = _get_collection(client)

    if video_id:
        scoped = collection.get(where={"video_id": video_id}, include=[])
        scoped_count = len(scoped["ids"])
        if scoped_count == 0:
            logger.warning("No embeddings found for video_id=%s", video_id)
            return []
        actual_k = min(top_k, scoped_count)
        where_clause = {"video_id": video_id}
    else:
        total_count = collection.count()
        if total_count == 0:
            return []
        actual_k = min(top_k, total_count)
        where_clause = None

    query_embedding = _embed([query])[0]

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": actual_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_clause:
        query_kwargs["where"] = where_clause

    results = collection.query(**query_kwargs)

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


def delete_video_embeddings(video_id: str) -> int:
    client = _get_client()
    collection = _get_collection(client)

    existing = collection.get(where={"video_id": video_id}, include=[])
    ids_to_delete = existing["ids"]

    if not ids_to_delete:
        logger.info("No embeddings found for video_id=%s — nothing deleted", video_id)
        return 0

    collection.delete(ids=ids_to_delete)
    logger.info("Deleted %d embeddings for video_id=%s", len(ids_to_delete), video_id)
    return len(ids_to_delete)


def clear_collection() -> None:
    client = _get_client()
    try:
        client.delete_collection(_COLLECTION_NAME)
        logger.info("ChromaDB collection '%s' dropped", _COLLECTION_NAME)
    except Exception:
        pass
