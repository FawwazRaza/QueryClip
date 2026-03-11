"""YouTube transcript extraction and LangChain Document creation."""

import concurrent.futures
import re
import time
import logging
from typing import List, Tuple

import yt_dlp
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    IpBlocked,
    RequestBlocked,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


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


def _parse_vtt(content: str) -> str:
    """Convert VTT/SRT subtitle content to plain text, deduplicating lines."""
    seen: set = set()
    text_parts: List[str] = []

    for line in content.strip().splitlines():
        line = line.strip()
        if not line or "-->" in line or line.startswith("WEBVTT") or line.isdigit():
            continue
        clean = re.sub(r"<[^>]+>", "", line)
        clean = re.sub(r"\s+", " ", clean).strip()
        if clean and clean not in seen:
            seen.add(clean)
            text_parts.append(clean)

    return " ".join(text_parts)


def _fetch_via_yt_dlp(video_id: str) -> str:
    """Fetch transcript via yt-dlp subtitle extraction.

    Tries multiple player clients (ios, android, web) with Node.js runtime
    enabled to bypass YouTube's PO-token requirement.  Subtitle content is
    fetched in-memory from the subtitle URL — no files are written to disk.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    preferred_langs = ["en", "en-US", "en-GB"]

    # Try multiple player client strategies in order of reliability
    strategies = [
        ["ios", "android"],
        ["web"],
        ["ios"],
    ]

    for player_clients in strategies:
        ydl_opts = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
            "js_runtimes": "node",  # Use Node.js to extract PO tokens
            "extractor_args": {
                "youtube": {
                    "player_client": player_clients,
                }
            },
            "http_headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            },
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                subtitles = info.get("subtitles", {})
                auto_captions = info.get("automatic_captions", {})

                # Prefer manual subtitles; fall back to auto-generated
                for source in [subtitles, auto_captions]:
                    if not source:
                        continue
                    lang_order = [l for l in preferred_langs if l in source] + [
                        l for l in source if l not in preferred_langs
                    ]
                    for lang in lang_order:
                        for fmt in source.get(lang, []):
                            if fmt.get("ext") == "vtt" and fmt.get("url"):
                                try:
                                    raw = ydl.urlopen(fmt["url"]).read().decode("utf-8")
                                    text = _parse_vtt(raw)
                                    if len(text) > 50:
                                        logger.info(
                                            "yt-dlp: subtitles for %s via %s (lang=%s)",
                                            video_id, player_clients, lang,
                                        )
                                        return text
                                except Exception:
                                    continue
        except Exception as exc:
            logger.warning(
                "yt-dlp strategy %s failed for %s: %s",
                player_clients, video_id, exc,
            )
            continue

    return ""


def _fetch_transcript(video_id: str, max_retries: int = 3) -> str:
    """Fetch full transcript text for a YouTube video.

    Layer 1 — youtube-transcript-api v1.x (instance-based): fast, low-bandwidth.
    Layer 2 — yt-dlp VTT extraction: robust, bypasses cloud/ngrok IP blocks.

    NOTE: youtube-transcript-api v1.x requires instantiation — static class
    methods from v0.x (e.g. list_transcripts) no longer exist.
    """
    # ── Layer 1: youtube-transcript-api ──────────────────────────────────────
    for attempt in range(max_retries):
        try:
            api = YouTubeTranscriptApi()

            # Direct fetch — most efficient path when language is known
            try:
                fetched = api.fetch(video_id, languages=["en", "en-US", "en-GB"])
                text = " ".join(s.text.strip() for s in fetched if s.text)
                if text:
                    logger.info("Transcript API: direct fetch for %s", video_id)
                    return text
            except NoTranscriptFound:
                pass  # Fall through to discovery mode

            # Discovery mode — list all available transcripts and pick best
            transcript_list = api.list(video_id)

            # Manual captions (highest quality)
            for lang in ["en", "en-US", "en-GB"]:
                try:
                    t = transcript_list.find_manually_created_transcript([lang])
                    ft = t.fetch()
                    text = " ".join(s.text.strip() for s in ft if s.text)
                    if text:
                        logger.info("Transcript API: manual %s for %s", lang, video_id)
                        return text
                except NoTranscriptFound:
                    continue

            # Auto-generated captions
            for lang in ["en", "en-US", "en-GB"]:
                try:
                    t = transcript_list.find_generated_transcript([lang])
                    ft = t.fetch()
                    text = " ".join(s.text.strip() for s in ft if s.text)
                    if text:
                        logger.info("Transcript API: auto-gen %s for %s", lang, video_id)
                        return text
                except NoTranscriptFound:
                    continue

            # Any available language as last resort
            available = list(transcript_list)
            if available:
                ft = available[0].fetch()
                text = " ".join(s.text.strip() for s in ft if s.text)
                if text:
                    logger.info("Transcript API: fallback lang for %s", video_id)
                    return text

            # Transcripts visible but none returned text — no point retrying
            break

        except (TranscriptsDisabled, VideoUnavailable, IpBlocked, RequestBlocked):
            logger.info(
                "Transcript API: blocked/disabled for %s — falling back to yt-dlp",
                video_id,
            )
            break  # Jump straight to Layer 2
        except Exception as exc:
            logger.warning(
                "Transcript API attempt %d/%d for %s: %s",
                attempt + 1, max_retries, video_id, exc,
            )
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    # ── Layer 2: yt-dlp VTT extraction ───────────────────────────────────────
    logger.info("Falling back to yt-dlp for %s", video_id)
    return _fetch_via_yt_dlp(video_id)


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
    failed_titles: List[str] = []

    def _fetch_one(video: dict) -> Tuple[dict, str]:
        """Fetch transcript for a single video; returns (video, text)."""
        return video, _fetch_transcript(video["video_id"])

    # For playlists use a thread pool; for a single video keep it simple.
    if len(videos) == 1:
        pairs = [_fetch_one(videos[0])]
    else:
        max_workers = min(4, len(videos))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            pairs = list(pool.map(_fetch_one, videos))

    for video, full_text in pairs:
        if not full_text or len(full_text.strip()) < 50:
            failed_titles.append(video["title"])
            logger.warning("No usable transcript for: %s (%s)", video["title"], video["video_id"])
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
        failed = ", ".join(failed_titles) if failed_titles else "all videos"
        raise ValueError(
            f"No transcripts could be fetched for: {failed}. "
            "Both youtube-transcript-api and yt-dlp subtitle extraction failed. "
            "Possible causes: captions disabled on the video, age-gated content, "
            "or the IP is temporarily rate-limited by YouTube."
        )

    return all_documents


def get_video_titles(url: str) -> List[str]:
    """Return the titles of all videos at a given URL without fetching transcripts."""
    return [v["title"] for v in _extract_video_metadata(url)]
