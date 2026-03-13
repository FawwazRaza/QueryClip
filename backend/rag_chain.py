import json
import os
import re
from typing import Generator, List, Optional

from dotenv import load_dotenv
from groq import Groq

from backend.vector_store import query_documents

load_dotenv()

_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
_MODEL = "llama-3.3-70b-versatile"
_MAX_HISTORY_TURNS = 6

_SYSTEM_PROMPT = """\
You are QueryClip, a knowledgeable and conversational video assistant. You have access \
to transcripts of YouTube videos and help users deeply understand and explore the content \
through natural, engaging conversation.

Your personality:
- Warm, enthusiastic, and genuinely helpful — like a friend who watched the video
- Reference specific details and examples from the transcript to support every answer
- Proactively mention related points from the video that might interest the user
- Use clear, conversational language; format longer answers with bullet points or numbered steps
- When the video uses the speaker's exact words or a memorable phrase, quote it briefly

Strict rules:
1. Answer ONLY using the provided transcript excerpts. Never invent facts, statistics, names, or dates.
2. If the answer is not in the excerpts, say exactly:
   "That topic isn't covered in the videos I have access to. Is there something else from the content I can help you with?"
3. For partial matches, share what the transcript does cover and note what's missing.
4. Attribute information to the video naturally: "The speaker explains...", "In the video, they mention..."
5. Never show <think> tags, meta-commentary, or internal reasoning."""

_GREETING_PROMPT = """\
You are QueryClip, a friendly YouTube video transcript assistant. You help users \
understand and explore video content through natural conversation.

Be warm, concise, and welcoming. If the user is just saying hello or asking what you can do, \
tell them they can paste any YouTube video or playlist URL in the sidebar to get started, \
then ask questions about the content. Never fabricate information."""

_ROUTER_PROMPT = (
    "Classify the user message into exactly one category. Respond with ONE WORD only.\n\n"
    "DEFAULT — Greeting, farewell, small talk, or asking what you can do.\n"
    "UNSAFE  — Harmful, offensive, illegal, or inappropriate content.\n"
    "BOT     — Any question, topic request, or information-seeking about video content.\n\n"
    "Respond with only: DEFAULT, UNSAFE, or BOT"
)


def _clean(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _route_query(query: str, client: Groq) -> str:
    try:
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _ROUTER_PROMPT},
                {"role": "user", "content": query},
            ],
            max_tokens=5,
            temperature=0.0,
            timeout=8,
        )
        raw = _clean(response.choices[0].message.content).upper()
        first_word = raw.split()[0] if raw.split() else "BOT"
        return first_word if first_word in {"DEFAULT", "UNSAFE", "BOT"} else "BOT"
    except Exception:
        return "BOT"


def _build_context(chunks: List[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("metadata", {}).get("title", "Unknown Video")
        pct = round(chunk.get("similarity", 0) * 100)
        parts.append(f'[Excerpt {i} from "{title}" \u2014 {pct}% relevance]\n{chunk["text"]}')
    return "\n\n---\n\n".join(parts)


def _build_rag_messages(
    query: str,
    context: str,
    chat_history: Optional[List[dict]],
) -> List[dict]:
    messages: List[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    if chat_history:
        for msg in chat_history[-_MAX_HISTORY_TURNS:]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

    messages.append({
        "role": "user",
        "content": (
            f"Here are the relevant transcript excerpts from the videos:\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"My question: {query}"
        ),
    })
    return messages


def _build_greeting_messages(
    query: str,
    chat_history: Optional[List[dict]],
) -> List[dict]:
    messages: List[dict] = [{"role": "system", "content": _GREETING_PROMPT}]
    if chat_history:
        for msg in chat_history[-4:]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": query})
    return messages


def _sources_from_chunks(chunks: List[dict]) -> List[dict]:
    return [
        {
            "title": c.get("metadata", {}).get("title", "Unknown"),
            "url": c.get("metadata", {}).get("url", ""),
            "similarity": c.get("similarity", 0),
            "text_preview": c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
        }
        for c in chunks
    ]


def _stream_tokens_clean(stream) -> Generator[str, None, None]:
    in_think = False
    pending = ""

    for chunk in stream:
        token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
        if not token:
            continue
        pending += token

        while pending:
            if in_think:
                end = pending.find("</think>")
                if end == -1:
                    pending = ""
                    break
                pending = pending[end + 8:]
                in_think = False
            else:
                start = pending.find("<think>")
                if start == -1:
                    yield pending
                    pending = ""
                    break
                if start > 0:
                    yield pending[:start]
                pending = pending[start + 7:]
                in_think = True


def answer_query(
    query: str,
    chat_history: Optional[List[dict]] = None,
    top_k: int = 6,
    video_id: Optional[str] = None,
) -> dict:
    client = Groq(api_key=_GROQ_API_KEY)
    route = _route_query(query, client)

    if route == "UNSAFE":
        return {
            "answer": "I'm not able to help with that kind of request.",
            "sources": [],
            "route": route,
        }

    if route == "DEFAULT":
        messages = _build_greeting_messages(query, chat_history)
        try:
            r = client.chat.completions.create(
                model=_MODEL,
                messages=messages,
                max_tokens=400,
                temperature=0.5,
                timeout=15,
            )
            return {
                "answer": _clean(r.choices[0].message.content),
                "sources": [],
                "route": route,
            }
        except Exception:
            return {
                "answer": (
                    "Hi! I'm QueryClip, your YouTube video assistant. "
                    "Paste a YouTube URL in the sidebar and ask me anything about the content."
                ),
                "sources": [],
                "route": route,
            }

    chunks = query_documents(query, top_k=top_k, video_id=video_id)

    if not chunks:
        if video_id:
            return {
                "answer": (
                    "This video has not been indexed yet or its embeddings were removed. "
                    "Please re-process the video URL to index its transcript."
                ),
                "sources": [],
                "route": route,
            }
        return {
            "answer": (
                "No videos have been indexed yet. "
                "Paste a YouTube URL in the sidebar and click Process to get started."
            ),
            "sources": [],
            "route": route,
        }

    relevant = [c for c in chunks if c.get("similarity", 0) > 0.2]
    if not relevant:
        return {
            "answer": (
                "That topic doesn't appear to be covered in the indexed videos. "
                "Try rephrasing your question, or ask about something else from the content."
            ),
            "sources": [],
            "route": route,
        }

    context = _build_context(relevant)
    messages = _build_rag_messages(query, context, chat_history)

    try:
        r = client.chat.completions.create(
            model=_MODEL,
            messages=messages,
            max_tokens=1200,
            temperature=0.15,
            timeout=45,
        )
        answer = _clean(r.choices[0].message.content)
    except Exception as exc:
        answer = f"I encountered an error while generating a response: {exc}"

    return {
        "answer": answer,
        "sources": _sources_from_chunks(relevant),
        "route": route,
    }


def stream_answer_query(
    query: str,
    chat_history: Optional[List[dict]] = None,
    top_k: int = 6,
    video_id: Optional[str] = None,
) -> Generator[str, None, None]:
    client = Groq(api_key=_GROQ_API_KEY)
    route = _route_query(query, client)

    if route == "UNSAFE":
        yield json.dumps({"type": "sources", "sources": [], "route": route})
        yield json.dumps({"type": "token", "token": "I'm not able to help with that kind of request."})
        return

    if route == "DEFAULT":
        yield json.dumps({"type": "sources", "sources": [], "route": route})
        messages = _build_greeting_messages(query, chat_history)
        try:
            stream = client.chat.completions.create(
                model=_MODEL,
                messages=messages,
                max_tokens=400,
                temperature=0.5,
                stream=True,
            )
            for token in _stream_tokens_clean(stream):
                yield json.dumps({"type": "token", "token": token})
        except Exception:
            yield json.dumps({
                "type": "token",
                "token": (
                    "Hi! I'm QueryClip, your YouTube video assistant. "
                    "Paste a YouTube URL in the sidebar to get started."
                ),
            })
        return

    chunks = query_documents(query, top_k=top_k, video_id=video_id)

    if not chunks:
        yield json.dumps({"type": "sources", "sources": [], "route": route})
        if video_id:
            yield json.dumps({
                "type": "token",
                "token": (
                    "This video has not been indexed yet or its embeddings were removed. "
                    "Please re-process the video URL to index its transcript."
                ),
            })
        else:
            yield json.dumps({
                "type": "token",
                "token": (
                    "No videos have been indexed yet. "
                    "Paste a YouTube URL in the sidebar and click Process to get started."
                ),
            })
        return

    relevant = [c for c in chunks if c.get("similarity", 0) > 0.2]

    yield json.dumps({
        "type": "sources",
        "sources": _sources_from_chunks(relevant),
        "route": route,
    })

    if not relevant:
        yield json.dumps({
            "type": "token",
            "token": (
                "That topic doesn't appear to be covered in the indexed videos. "
                "Try rephrasing your question, or ask about something else from the content."
            ),
        })
        return

    context = _build_context(relevant)
    messages = _build_rag_messages(query, context, chat_history)

    try:
        stream = client.chat.completions.create(
            model=_MODEL,
            messages=messages,
            max_tokens=1200,
            temperature=0.15,
            stream=True,
        )
        for token in _stream_tokens_clean(stream):
            yield json.dumps({"type": "token", "token": token})
    except Exception as exc:
        yield json.dumps({"type": "error", "message": str(exc)})
