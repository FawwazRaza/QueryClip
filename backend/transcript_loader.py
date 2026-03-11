"""YouTube transcript extraction and LangChain Document creation."""

import time
from typing import List

import yt_dlp
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _extract_video_metadata(url: str) -> List[dict]:
    """Extract video metadata from a YouTube URL using yt-dlp.

    Args:
        url: YouTube video or playlist URL.

    Returns:
        List of dicts with video_id, title, url, playlist_name.
    """
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "no_warnings": True,
        "socket_timeout": 10,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if not info:
        return []

    videos: List[dict] = []

    if info.get("_type") == "playlist":
        playlist_name = info.get("title", "Unknown Playlist")
        for entry in info.get("entries") or []:
            if entry and entry.get("id"):
                videos.append(
                    {
                        "video_id": entry["id"],
                        "title": entry.get("title", "Unknown"),
                        "url": f"https://www.youtube.com/watch?v={entry['id']}",
                        "playlist_name": playlist_name,
                    }
                )
    else:
        if info.get("id"):
            videos.append(
                {
                    "video_id": info["id"],
                    "title": info.get("title", "Unknown"),
                    "url": url,
                    "playlist_name": "",
                }
            )

    return videos


def _fetch_transcript(video_id: str, max_retries: int = 3) -> List[dict]:
    """Fetch transcript segments for a YouTube video.

    Tries manual captions first, then auto-generated, then any available.

    Args:
        video_id: YouTube video ID.
        max_retries: Retry count on transient failures.

    Returns:
        List of segment dicts with keys: text, start, duration.
    """
    for attempt in range(max_retries):
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            for strategy in ["manual", "generated"]:
                for lang in ["en", "en-US", "en-GB"]:
                    try:
                        if strategy == "manual":
                            t = transcript_list.find_manually_created_transcript([lang])
                        else:
                            t = transcript_list.find_generated_transcript([lang])
                        return t.fetch()
                    except NoTranscriptFound:
                        continue

            available = list(transcript_list)
            if available:
                return available[0].fetch()

            return []

        except TranscriptsDisabled:
            return []
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                return []

    return []


def load_youtube_documents(
    url: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[Document]:
    """Process a YouTube URL and return chunked LangChain Documents.

    Args:
        url: YouTube video or playlist URL.
        chunk_size: Characters per text chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        List of Document objects ready for embedding.

    Raises:
        ValueError: If the URL is invalid or no transcripts could be fetched.
    """
    videos = _extract_video_metadata(url)
    if not videos:
        raise ValueError(f"Could not extract video information from: {url}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    all_documents: List[Document] = []

    for video in videos:
        segments = _fetch_transcript(video["video_id"])
        if not segments:
            continue

        full_text = " ".join(
            seg.get("text", "").strip() for seg in segments if seg.get("text")
        )
        if not full_text.strip():
            continue

        chunks = splitter.split_text(full_text)

        for index, chunk in enumerate(chunks):
            all_documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "title": video["title"],
                        "url": video["url"],
                        "video_id": video["video_id"],
                        "playlist_name": video.get("playlist_name", ""),
                        "chunk_index": index,
                    },
                )
            )

    if not all_documents:
        raise ValueError(
            "No transcripts could be fetched. "
            "The video(s) may have transcripts disabled or be unavailable."
        )

    return all_documents


def get_video_titles(url: str) -> List[str]:
    """Return the titles of all videos at a given URL without fetching transcripts.

    Args:
        url: YouTube video or playlist URL.

    Returns:
        List of video title strings.
    """
    return [v["title"] for v in _extract_video_metadata(url)]
