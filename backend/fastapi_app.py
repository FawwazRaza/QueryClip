import asyncio
import json
import queue
import sys
import threading
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.transcript_loader import load_youtube_documents
from backend.vector_store import (
    add_documents,
    list_indexed_videos,
    clear_collection,
    delete_video_embeddings,
)
from backend.rag_chain import answer_query, stream_answer_query

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
    top_k: int = Field(default=6, ge=1, le=10)
    video_id: Optional[str] = Field(default=None, description="Filter answers to a specific video")


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "QueryClip API is running"}


@app.post("/process")
async def process_url(request: ProcessRequest):
    url = request.url.strip()

    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty.")

    if "youtube.com" not in url and "youtu.be" not in url:
        raise HTTPException(status_code=400, detail="URL must be a valid YouTube link.")

    try:
        documents = await asyncio.to_thread(load_youtube_documents, url)
        count = await asyncio.to_thread(add_documents, documents)
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
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = answer_query(
        query=query,
        chat_history=request.chat_history,
        top_k=request.top_k,
        video_id=request.video_id or None,
    )
    return JSONResponse(content=result)


@app.get("/videos")
async def list_videos():
    videos = list_indexed_videos()
    return JSONResponse(content={"videos": videos, "count": len(videos)})


@app.delete("/videos/{video_id}")
async def delete_video(video_id: str):
    deleted_count = await asyncio.to_thread(delete_video_embeddings, video_id)
    if deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No embeddings found for video_id '{video_id}'.",
        )
    return JSONResponse(
        content={
            "success": True,
            "video_id": video_id,
            "deleted_chunks": deleted_count,
            "message": f"Removed {deleted_count} chunks for video '{video_id}'.",
        }
    )


@app.delete("/videos")
async def clear_videos():
    await asyncio.to_thread(clear_collection)
    return JSONResponse(
        content={"success": True, "message": "All indexed videos have been cleared."}
    )


@app.post("/query/stream")
async def stream_query_endpoint(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    q: queue.Queue = queue.Queue()
    _SENTINEL = object()

    def _producer() -> None:
        try:
            for event_json in stream_answer_query(
                query=query,
                chat_history=request.chat_history,
                top_k=request.top_k,
                video_id=request.video_id or None,
            ):
                q.put(event_json)
        finally:
            q.put(_SENTINEL)

    threading.Thread(target=_producer, daemon=True).start()

    async def _sse_generator() -> AsyncGenerator[str, None]:
        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is _SENTINEL:
                break
            yield f"data: {item}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
