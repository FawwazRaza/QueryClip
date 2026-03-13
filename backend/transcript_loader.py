import concurrent.futures
import logging
import re
import time
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

_TRANSCRIPT_API_TIMEOUT = 90
_YTDLP_SOCKET_TIMEOUT = 30


def _extract_video_metadata(url: str) -> List[dict]:
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "no_warnings": True,
        "socket_timeout": 15,
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
    url = f"https://www.youtube.com/watch?v={video_id}"
    preferred_langs = ["en", "en-US", "en-GB"]

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
            "socket_timeout": _YTDLP_SOCKET_TIMEOUT,
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
                                            "yt-dlp subtitles for %s via %s (lang=%s)",
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


def _call_transcript_api(video_id: str) -> str:
    api = YouTubeTranscriptApi()

    try:
        fetched = api.fetch(video_id, languages=["en", "en-US", "en-GB"])
        text = " ".join(s.text.strip() for s in fetched if s.text)
        if text:
            logger.info("Transcript API: direct fetch for %s", video_id)
            return text
    except NoTranscriptFound:
        pass

    transcript_list = api.list(video_id)

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

    available = list(transcript_list)
    if available:
        ft = available[0].fetch()
        text = " ".join(s.text.strip() for s in ft if s.text)
        if text:
            logger.info("Transcript API: fallback lang for %s", video_id)
            return text

    return ""


def _fetch_transcript(video_id: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call_transcript_api, video_id)
                try:
                    result = future.result(timeout=_TRANSCRIPT_API_TIMEOUT)
                    if result:
                        return result
                    break
                except concurrent.futures.TimeoutError:
                    logger.warning(
                        "Transcript API timed out after %ds for %s (attempt %d/%d)",
                        _TRANSCRIPT_API_TIMEOUT, video_id, attempt + 1, max_retries,
                    )
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    continue

        except (TranscriptsDisabled, VideoUnavailable, IpBlocked, RequestBlocked) as exc:
            logger.info(
                "Transcript API blocked/disabled for %s (%s) — falling back to yt-dlp",
                video_id, type(exc).__name__,
            )
            break
        except Exception as exc:
            logger.warning(
                "Transcript API attempt %d/%d for %s: %s",
                attempt + 1, max_retries, video_id, exc,
            )
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    logger.info("Falling back to yt-dlp for %s", video_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_fetch_via_yt_dlp, video_id)
        try:
            return future.result(timeout=_TRANSCRIPT_API_TIMEOUT * 2)
        except concurrent.futures.TimeoutError:
            logger.error(
                "yt-dlp also timed out after %ds for %s",
                _TRANSCRIPT_API_TIMEOUT * 2, video_id,
            )
            return ""


def load_youtube_documents(
    url: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[Document]:
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
        return video, _fetch_transcript(video["video_id"])

    if len(videos) == 1:
        pairs = [_fetch_one(videos[0])]
    else:
        max_workers = min(4, len(videos))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            pairs = list(pool.map(_fetch_one, videos))

    for video, full_text in pairs:
        if not full_text or len(full_text.strip()) < 50:
            failed_titles.append(video["title"])
            logger.warning(
                "Transcript extraction failed for: %s (%s)",
                video["title"], video["video_id"],
            )
            continue

        logger.info(
            "Transcript fetched for %s — %d characters", video["video_id"], len(full_text)
        )

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
            f"Transcript extraction failed for: {failed}. "
            "Both youtube-transcript-api and yt-dlp subtitle extraction failed. "
            "Possible causes: captions disabled, age-gated content, "
            "IP rate-limited by YouTube, or the video is too new."
        )

    logger.info(
        "Loaded %d chunks from %d video(s)", len(all_documents), len(videos) - len(failed_titles)
    )
    return all_documents


def get_video_titles(url: str) -> List[str]:
    return [v["title"] for v in _extract_video_metadata(url)]
