import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from retriever import ChromaRetriever
from groq import Groq
import re

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_llm_response(context, question, chat_history):
    client = Groq(api_key=GROQ_API_KEY)
    model_name = "deepseek-r1-distill-llama-70b"

    history_str = ""
    for msg in chat_history:
        if msg["role"] == "user":
            history_str += f"User: {msg['content']}\n"
        else:
            history_str += f"Assistant: {msg['content']}\n"
    # history_str = ""
    # for msg in chat_history:
    #     role = "User" if msg["role"] == "user" else "Assistant"
    #     history_str += f"{role}: {msg['content']}\n"
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the question using ONLY the provided context and chat history. "
                "Do NOT show your internal reasoning, thoughts, or thinking process to the user. "
                "If the answer is not present in the context, reply with 'Not found in the dataset.'"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Chat History:\n{history_str}\n"
                f"Current Question: {question}\nAnswer:"
            ),
        },
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model_name,
    )
    answer = chat_completion.choices[0].message.content.strip()
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    return answer

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    chat_history: list = []

# @app.post("/query")
# async def query_endpoint(request: QueryRequest):
#     TOP_K = 3
#     retriever_chain = ChromaRetriever().as_retriever(search_kwargs={"k": TOP_K})
#     top_chunks = retriever_chain(request.query, k=TOP_K)
#     non_empty_chunks = [chunk for chunk in top_chunks if chunk["text"].strip()]
#     if not non_empty_chunks:
#         return JSONResponse(content={
#             "answer": "Not found in the dataset.",
#             "chunks": []
#         })
#     context = "\n\n".join(
#         f"[{i+1}] {chunk['text']} (Start: {chunk['start_time']}s, End: {chunk['end_time']}s)"
#         for i, chunk in enumerate(non_empty_chunks)
#     )
#     answer = get_llm_response(context, request.query, request.chat_history)
#     chunk_payload = [
#         {
#             "text": chunk["text"],
#             "start_time": chunk["start_time"],
#             "end_time": chunk["end_time"]
#         }
#         for chunk in non_empty_chunks
#     ]
#     return JSONResponse(content={
#         "answer": answer,
#         "chunks": chunk_payload
#     })
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    TOP_K = 3
    retriever_chain = ChromaRetriever().as_retriever(search_kwargs={"k": TOP_K})
    top_chunks = retriever_chain(request.query, k=TOP_K)
    non_empty_chunks = [chunk for chunk in top_chunks if chunk["text"].strip()]
    if not non_empty_chunks:
        return JSONResponse(content={
            "answer": "Not found in the dataset.",
            "chunks": []
        })
    context = "\n\n".join(
        f"[{i+1}] {chunk['text']} (File: {chunk['file_name']}, Start: {chunk['start_time']}s, End: {chunk['end_time']}s)"
        for i, chunk in enumerate(non_empty_chunks)
    )
    answer = get_llm_response(context, request.query, request.chat_history)
    chunk_payload = [
        {
            "text": chunk["text"],
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "file_name": chunk["file_name"]
        }
        for chunk in non_empty_chunks
    ]
    return JSONResponse(content={
        "answer": answer,
        "chunks": chunk_payload
    })
