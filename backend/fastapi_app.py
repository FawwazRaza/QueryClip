"""FastAPI backend for the QueryClip YouTube RAG system."""

import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Ensure project root is on the path when running via uvicorn
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.transcript_loader import load_youtube_documents
from backend.vector_store import add_documents, list_indexed_videos, clear_collection
from backend.rag_chain import answer_query

app = FastAPI(title="QueryClip API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessRequest(BaseModel):
    url: str = Field(..., description="YouTube video or playlist URL")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    chat_history: List[dict] = Field(default_factory=list)
    top_k: int = Field(default=4, ge=1, le=10)


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "QueryClip API is running"}


@app.post("/process")
async def process_url(request: ProcessRequest):
    """Extract transcripts from a YouTube URL and store chunks in ChromaDB."""
    url = request.url.strip()

    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty.")

    if "youtube.com" not in url and "youtu.be" not in url:
        raise HTTPException(status_code=400, detail="URL must be a valid YouTube link.")

    try:
        documents = load_youtube_documents(url)
        count = add_documents(documents)
        return JSONResponse(
            content={
                "success": True,
                "chunks_stored": count,
                "message": f"Successfully indexed {count} transcript chunks.",
            }
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Query the RAG system with a natural language question."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = answer_query(
        query=query,
        chat_history=request.chat_history,
        top_k=request.top_k,
    )
    return JSONResponse(content=result)


@app.get("/videos")
async def list_videos():
    """List all videos currently indexed in ChromaDB."""
    videos = list_indexed_videos()
    return JSONResponse(content={"videos": videos, "count": len(videos)})


@app.delete("/videos")
async def clear_videos():
    """Remove all indexed data from ChromaDB."""
    clear_collection()
    return JSONResponse(
        content={"success": True, "message": "All indexed videos have been cleared."}
    )
