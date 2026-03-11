"""QueryClip — YouTube Transcript RAG Chatbot."""

import json
import os
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(
    page_title="QueryClip",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Backend URL resolution ───────────────────────────────────────────────────

_STATIC_DOMAIN = "https://great-repeatedly-alien.ngrok-free.app"
_NGROK_HEADERS = {"ngrok-skip-browser-warning": "true"}


def _resolve_api_url() -> str:
    try:
        url = st.secrets.get("BACKEND_URL", "")
        if url:
            return url.rstrip("/")
    except Exception:
        pass

    env_url = os.getenv("BACKEND_URL", "")
    if env_url:
        return env_url.rstrip("/")

    saved = Path("ngrok_url.txt")
    if saved.exists():
        text = saved.read_text().strip()
        if text:
            return text.rstrip("/")

    return _STATIC_DOMAIN


API_URL = _resolve_api_url()

# ─── API helpers ──────────────────────────────────────────────────────────────


def _is_backend_available() -> bool:
    try:
        r = requests.get(f"{API_URL}/", timeout=5, headers=_NGROK_HEADERS)
        return r.status_code == 200
    except Exception:
        return False


def _process_url(youtube_url: str) -> dict:
    r = requests.post(
        f"{API_URL}/process",
        json={"url": youtube_url},
        timeout=300,
        headers=_NGROK_HEADERS,
    )
    r.raise_for_status()
    return r.json()


def _stream_query_backend(query: str, history: list):
    """Stream the answer from the /query/stream SSE endpoint.

    Returns a ``(token_generator, sources_holder)`` tuple where:
    - ``token_generator`` is a generator of raw answer string tokens suitable
      for ``st.write_stream()``.
    - ``sources_holder`` is a mutable list; sources are appended to it as a
      side-effect while the generator is consumed.
    """
    sources_holder: list = []

    def _token_gen():
        with requests.post(
            f"{API_URL}/query/stream",
            json={"query": query, "chat_history": history},
            stream=True,
            timeout=90,
            headers=_NGROK_HEADERS,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line.startswith("data: "):
                    continue
                payload = raw_line[6:]
                if payload == "[DONE]":
                    return
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "sources":
                    sources_holder.extend(event.get("sources", []))
                elif event.get("type") == "token":
                    yield event["token"]
                elif event.get("type") == "error":
                    raise RuntimeError(event.get("message", "Unknown streaming error"))

    return _token_gen(), sources_holder


def _query_backend(query: str, history: list) -> dict:
    r = requests.post(
        f"{API_URL}/query",
        json={"query": query, "chat_history": history},
        timeout=30,
        headers=_NGROK_HEADERS,
    )
    r.raise_for_status()
    return r.json()


def _list_videos() -> list:
    try:
        r = requests.get(f"{API_URL}/videos", timeout=10, headers=_NGROK_HEADERS)
        if r.status_code == 200:
            return r.json().get("videos", [])
    except Exception:
        pass
    return []


def _clear_all() -> None:
    requests.delete(f"{API_URL}/videos", timeout=10, headers=_NGROK_HEADERS)


# ─── Session state ────────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_urls" not in st.session_state:
    st.session_state.processed_urls: list = []
if "backend_ok" not in st.session_state:
    st.session_state.backend_ok = False

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("QueryClip")
    st.caption("YouTube Transcript RAG System")
    st.divider()

    col_check, col_status = st.columns([2, 1])
    with col_check:
        if st.button("Check Connection", use_container_width=True):
            st.session_state.backend_ok = _is_backend_available()
    with col_status:
        if st.session_state.backend_ok:
            st.success("Online")
        else:
            st.error("Offline")

    if not st.session_state.backend_ok:
        st.caption(f"Expected at: {API_URL}")

    st.divider()

    st.subheader("Add YouTube Content")
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://youtube.com/watch?v=... or playlist",
        label_visibility="collapsed",
    )

    col_proc, col_clr = st.columns(2)
    with col_proc:
        process_btn = st.button("Process", use_container_width=True, type="primary")
    with col_clr:
        clear_btn = st.button("Clear All", use_container_width=True)

    if process_btn:
        if not youtube_url.strip():
            st.warning("Enter a YouTube URL first.")
        elif not st.session_state.backend_ok:
            st.error("Backend is offline. Start ngrok_backend.py, then click Check Connection.")
        else:
            with st.spinner("Extracting and indexing transcripts..."):
                try:
                    result = _process_url(youtube_url.strip())
                    chunks = result.get("chunks_stored", 0)
                    st.success(f"Indexed {chunks} transcript chunks.")
                    if youtube_url not in st.session_state.processed_urls:
                        st.session_state.processed_urls.append(youtube_url)
                    st.rerun()
                except requests.HTTPError as exc:
                    try:
                        detail = exc.response.json().get("detail", str(exc))
                    except Exception:
                        detail = str(exc)
                    st.error(f"Processing failed: {detail}")
                except requests.ConnectionError:
                    st.error("Cannot reach backend. Is ngrok_backend.py running?")
                except Exception as exc:
                    st.error(f"Error: {exc}")

    if clear_btn:
        if not st.session_state.backend_ok:
            st.error("Backend is offline.")
        else:
            _clear_all()
            st.session_state.processed_urls = []
            st.session_state.chat_history = []
            st.success("All indexed data cleared.")
            st.rerun()

    st.divider()

    st.subheader("Indexed Videos")
    if st.session_state.backend_ok:
        videos = _list_videos()
        if videos:
            for v in videos:
                with st.expander(v.get("title", "Unknown"), expanded=False):
                    if v.get("url"):
                        st.markdown(f"[Open on YouTube]({v['url']})")
                    if v.get("playlist_name"):
                        st.caption(f"Playlist: {v['playlist_name']}")
        else:
            st.caption("No videos indexed yet.")
    else:
        st.caption("Connect to backend to view indexed videos.")

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.caption(f"Backend: {API_URL}")

# ─── Main area ────────────────────────────────────────────────────────────────

st.title("QueryClip")
st.caption("Ask questions grounded in the content of any YouTube video or playlist.")

if not st.session_state.chat_history:
    st.info(
        "Add a YouTube URL in the sidebar and click Process. "
        "Once indexed, ask any question and receive answers drawn strictly from the transcript."
    )

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("View sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src.get('title', 'Unknown')}** — relevance: {src.get('similarity', 0):.2f}"
                    )
                    if src.get("url"):
                        st.caption(src["url"])
                    if src.get("text_preview"):
                        st.caption(f'"{src["text_preview"]}"')
                    st.divider()

user_input = st.chat_input("Ask a question about your videos...")

if user_input:
    stripped = user_input.strip()

    if stripped.lower() == "/clear":
        st.session_state.chat_history = []
        st.rerun()

    elif stripped.lower() == "/help":
        with st.chat_message("assistant"):
            st.markdown(
                "**Commands:**\n"
                "- `/clear` — Clear chat history\n"
                "- `/help` — Show this message\n\n"
                "**Usage:**\n"
                "1. Paste a YouTube URL in the sidebar, click Process.\n"
                "2. Wait for indexing to complete.\n"
                "3. Ask questions — answers come only from indexed transcripts."
            )

    else:
        with st.chat_message("user"):
            st.write(stripped)

        api_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history
        ]
        st.session_state.chat_history.append({"role": "user", "content": stripped})

        if not st.session_state.backend_ok:
            st.session_state.backend_ok = _is_backend_available()

        if not st.session_state.backend_ok:
            err = "Backend is not connected. Start ngrok_backend.py and click Check Connection."
            with st.chat_message("assistant"):
                st.error(err)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": err, "sources": []}
            )
        else:
            with st.chat_message("assistant"):
                try:
                    gen, sources_holder = _stream_query_backend(stripped, api_history)
                    answer = st.write_stream(gen)

                    if sources_holder:
                        with st.expander("View sources", expanded=False):
                            for src in sources_holder:
                                st.markdown(
                                    f"**{src.get('title', 'Unknown')}**"
                                    f" \u2014 relevance: {src.get('similarity', 0):.0%}"
                                )
                                if src.get("url"):
                                    st.caption(src["url"])
                                if src.get("text_preview"):
                                    st.caption(f'\u201c{src["text_preview"]}\u201d')
                                st.divider()

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "sources": list(sources_holder)}
                    )

                except requests.HTTPError as exc:
                    try:
                        detail = exc.response.json().get("detail", str(exc))
                    except Exception:
                        detail = str(exc)
                    err = f"Request failed: {detail}"
                    st.error(err)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": err, "sources": []}
                    )
                except requests.ConnectionError:
                    err = "Cannot reach backend."
                    st.error(err)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": err, "sources": []}
                    )
                except Exception as exc:
                    err = f"An error occurred: {exc}"
                    st.error(err)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": err, "sources": []}
                    )

