import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from retriever import ChromaRetriever
from groq import Groq
import re

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set in your .env file. The application may not work correctly.")

app = FastAPI()

def get_llm_response(context, question, chat_history=""):
    client = Groq(api_key=GROQ_API_KEY)
    model_name = "deepseek-r1-distill-llama-70b"

    # Format chat history
    history_str = ""
    if isinstance(chat_history, list):
        for msg in chat_history:
            if msg.get("role") == "user":
                history_str += f"User: {msg.get('content', '')}\n"
            else:
                history_str += f"Assistant: {msg.get('content', '')}\n"
    else:
        history_str = chat_history
    
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

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_name,
        )
        answer = chat_completion.choices[0].message.content.strip()
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        return answer
    except Exception as e:
        print(f"Error in LLM response: {str(e)}")
        return "Sorry, I encountered an error processing your request."

def get_general_response(query, chat_history=""):
    client = Groq(api_key=GROQ_API_KEY)
    model_name = "deepseek-r1-distill-llama-70b"
    
    # Format chat history
    history_str = ""
    if isinstance(chat_history, list):
        for msg in chat_history:
            if msg.get("role") == "user":
                history_str += f"User: {msg.get('content', '')}\n"
            else:
                history_str += f"Assistant: {msg.get('content', '')}\n"
    else:
        history_str = chat_history
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant for a video browsing platform. "
                "The user is asking a general question not related to specific video content. "
                "Provide a helpful, conversational response."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Chat History:\n{history_str}\n"
                f"User Question: {query}\n"
                f"Please provide a helpful, conversational response:"
            ),
        },
    ]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_name,
        )
        answer = chat_completion.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error in general response: {str(e)}")
        return "Sorry, I encountered an error processing your request."

def is_video_specific_query(query, chat_history=""):
    client = Groq(api_key=GROQ_API_KEY)
    model_name = "deepseek-r1-distill-llama-70b"
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a classifier that determines if a query is asking about specific video content or not. "
                "Reply with ONLY 'yes' if the query is asking about information that might be in video content, "
                "or 'no' if it's a general query, greeting, or not related to video content."
            ),
        },
        {
            "role": "user",
            "content": f"Query: {query}\nIs this asking about specific video content? (yes/no)"
        },
    ]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_name,
        )
        answer = chat_completion.choices[0].message.content.strip().lower()
        return "yes" in answer
    except Exception as e:
        print(f"Error in classifier: {str(e)}")
        return True  # Default to retrieval on error

# Add a health check endpoint
@app.get("/")
async def health_check():
    return {"status": "ok", "message": "API is running"}

class QueryRequest(BaseModel):
    query: str
    chat_history: list = []

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        # Determine if query is about video content
        is_video_query = is_video_specific_query(request.query)
        
        # If it's a video query, use retrieval
        if is_video_query:
            # Get top chunks
            TOP_K = 3
            retriever_chain = ChromaRetriever().as_retriever(search_kwargs={"k": TOP_K})
            top_chunks = retriever_chain(request.query, k=TOP_K)
            non_empty_chunks = [chunk for chunk in top_chunks if chunk["text"].strip()]
            
            if not non_empty_chunks:
                return JSONResponse(content={
                    "answer": "Not found in the dataset.",
                    "source": None,
                    "chunks": []
                })
            
            # Get the top chunk based on similarity
            top_chunk = max(non_empty_chunks, key=lambda c: c.get("similarity", 0))
            
            # Create context from top chunk
            context = (
                f"{top_chunk['text']} "
                f"(File: {top_chunk['file_name']}, Start: {top_chunk['start_time']}s, End: {top_chunk['end_time']}s)"
            )
            
            # Get response from LLM
            answer = get_llm_response(context, request.query, request.chat_history)
            
            # Format chunks for response
            chunk_payload = [
                {
                    "text": chunk["text"],
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                    "file_name": chunk["file_name"],
                    "similarity": chunk.get("similarity", None)
                }
                for chunk in non_empty_chunks
            ]
            
            # Format source info
            source_info = {
                "file_name": top_chunk["file_name"],
                "start_time": top_chunk["start_time"],
                "end_time": top_chunk["end_time"]
            }
            
            return JSONResponse(content={
                "answer": answer,
                "source": source_info,
                "chunks": chunk_payload
            })
        else:
            # For general queries
            answer = get_general_response(request.query, request.chat_history)
            return JSONResponse(content={
                "answer": answer,
                "source": None,
                "chunks": []
            })
            
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
