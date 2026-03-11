"""RAG chain with Groq LLM and strict transcript-grounded responses."""

import os
import re
from typing import List, Optional

from dotenv import load_dotenv
from groq import Groq

from backend.vector_store import query_documents

load_dotenv()

_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
_RAG_MODEL = "llama-3.3-70b-versatile"
_ROUTER_MODEL = "llama-3.3-70b-versatile"

_SYSTEM_PROMPT = (
    "You are a video content assistant. "
    "Answer ONLY using the provided video transcript context. "
    "If the answer is not in the context, respond with exactly: "
    "'This information is not covered in the videos.' "
    "Do not invent facts, names, dates, or details absent from the transcript. "
    "Be concise and direct. Do not show internal reasoning steps."
)

_GREETING_PROMPT = (
    "You are a helpful video transcript chatbot. "
    "Respond briefly and professionally to greetings or capability questions. "
    "Do not fabricate information. Do not show reasoning steps."
)

_ROUTER_PROMPT = (
    "Classify the user query into exactly one category and respond with ONE WORD only.\n\n"
    "Categories:\n"
    "DEFAULT - Greeting, farewell, introduction, or question about bot capabilities.\n"
    "UNSAFE  - Offensive, illegal, sexual, violent, or harmful content.\n"
    "BOT     - Any substantive question or information request.\n\n"
    "Respond with only: DEFAULT, UNSAFE, or BOT. No punctuation, no explanation."
)


def _clean(text: str) -> str:
    """Remove think-block tags and trim whitespace."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _route_query(query: str, client: Groq) -> str:
    """Classify a query to determine the processing path."""
    try:
        response = client.chat.completions.create(
            model=_ROUTER_MODEL,
            messages=[
                {"role": "system", "content": _ROUTER_PROMPT},
                {"role": "user", "content": f"Query: {query}"},
            ],
            max_tokens=10,
            temperature=0.0,
            timeout=10,
        )
        raw = _clean(response.choices[0].message.content).upper()
        first_word = raw.split()[0] if raw.split() else "BOT"
        return first_word if first_word in {"DEFAULT", "UNSAFE", "BOT"} else "BOT"
    except Exception:
        return "BOT"


def _build_context(chunks: List[dict]) -> str:
    """Format retrieved chunks into an LLM-ready context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("metadata", {}).get("title", "Unknown")
        parts.append(f"[Source {i} — {title}]\n{chunk['text']}")
    return "\n\n".join(parts)


def _format_history(chat_history: Optional[List[dict]]) -> str:
    """Serialize the last three exchanges as plain text."""
    if not chat_history:
        return ""
    lines = []
    for msg in chat_history[-6:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


def answer_query(
    query: str,
    chat_history: Optional[List[dict]] = None,
    top_k: int = 4,
) -> dict:
    """Answer a user query using retrieval-augmented generation.

    Args:
        query: The user's question.
        chat_history: Previous conversation turns as list of role/content dicts.
        top_k: Number of transcript chunks to retrieve.

    Returns:
        Dict with keys: answer (str), sources (list), route (str).
    """
    client = Groq(api_key=_GROQ_API_KEY)
    route = _route_query(query, client)

    if route == "UNSAFE":
        return {
            "answer": "This request cannot be processed as it contains inappropriate content.",
            "sources": [],
            "route": route,
        }

    if route == "DEFAULT":
        history_str = _format_history(chat_history)
        user_msg = f"{history_str}\nUser: {query}" if history_str else query
        try:
            response = client.chat.completions.create(
                model=_RAG_MODEL,
                messages=[
                    {"role": "system", "content": _GREETING_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=300,
                temperature=0.3,
                timeout=10,
            )
            return {
                "answer": _clean(response.choices[0].message.content),
                "sources": [],
                "route": route,
            }
        except Exception:
            return {
                "answer": (
                    "Hello. I am a video transcript chatbot. "
                    "Add a YouTube URL in the sidebar and ask questions about the content."
                ),
                "sources": [],
                "route": route,
            }

    # BOT route — full RAG pipeline
    chunks = query_documents(query, top_k=top_k)

    if not chunks:
        return {
            "answer": (
                "No video content has been indexed yet. "
                "Please process a YouTube URL first."
            ),
            "sources": [],
            "route": route,
        }

    relevant = [c for c in chunks if c.get("similarity", 0) > 0.2]
    if not relevant:
        return {
            "answer": "This information is not covered in the videos.",
            "sources": [],
            "route": route,
        }

    context = _build_context(relevant)
    history_str = _format_history(chat_history)

    user_content = (
        f"Video Transcript Context:\n{context}\n\n"
        + (f"Previous conversation:\n{history_str}\n\n" if history_str else "")
        + f"Question: {query}\nAnswer:"
    )

    try:
        response = client.chat.completions.create(
            model=_RAG_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=800,
            temperature=0.1,
            timeout=30,
        )
        answer = _clean(response.choices[0].message.content)
    except Exception as exc:
        answer = f"An error occurred while generating the response: {exc}"

    sources = [
        {
            "title": c.get("metadata", {}).get("title", "Unknown"),
            "url": c.get("metadata", {}).get("url", ""),
            "similarity": c.get("similarity", 0),
            "text_preview": (
                c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"]
            ),
        }
        for c in relevant
    ]

    return {"answer": answer, "sources": sources, "route": route}
