import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import pyngrok.ngrok as ngrok
import uvicorn
import sys
import re
import asyncio
import json
from fastapi.responses import FileResponse
from pathlib import Path


# Import your existing modules
from retriever import ChromaRetriever
from groq import Groq


# VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/videos")
# os.makedirs(VIDEO_DIR, exist_ok=True)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set in your .env file. The application may not work correctly.")

if not NGROK_AUTH_TOKEN:
    print("WARNING: NGROK_AUTH_TOKEN is not set. Ngrok will have connection limitations.")
else:
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Create FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from Streamlit Cloud
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


class Route(BaseModel):
    destination: str = Field(..., description="Destination to route to")

def route_query(query: str) -> str:
    
    client = Groq(api_key=GROQ_API_KEY)
    model_name = "qwen-qwq-32b" 

    messages = [
        {
            "role": "system",
            "content": (
                "You are a classification assistant. Your task is to classify a user query into exactly one of the following routes:\n"
                "- 'DEFAULT': If the query is a greeting, farewell, or introduction about the chatbot. greetings like Hi, hello. Introduction like who are you? Introduce yourself? Why I use you etc choose this category\n"
                "- 'UNSAFE': If the query includes offensive, illegal, sexual, violent, or otherwise unsafe content.\n"
                "- 'Bot': For all queries not covered by the above categories.\n"
                "Respond with ONLY ONE WORD: DEFAULT, UNSAFE, or Bot."
                "You are a classification assistant. Respond with EXACTLY one word: DEFAULT, UNSAFE, or BOT.\n"
                 "Do NOT explain, reason, or think aloud. Just respond with one of the three words ONLY — no punctuation, no extra words."
                "Do NOT show your internal reasoning, thoughts, or thinking process to the user. "
                "Do not provide your <think> tag and process"

            ),
        },
        {
            "role": "user",
            "content": f"Query: {query}\nRoute:",
        },
    ]

    try:
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        classification = chat_completion.choices[0].message.content.strip().upper()
        classification = re.sub(r"<THINK>.*?</THINK>", "", classification, flags=re.DOTALL).strip()
        print(f"LLM classification: {classification}")

        if classification not in {"DEFAULT", "UNSAFE", "BOT"}:
            print(f"Unexpected LLM response: {classification}")
            return "Bot" 
        
        return classification
    except Exception as e:
        print(f"Error classifying query: {str(e)}")
        return "Bot"

async def stream_llm_response(context, question, chat_history=""):
    client = Groq(api_key=GROQ_API_KEY)
    model_name = "deepseek-r1-distill-llama-70b"

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
                "Do NOT show your internal reasoning, thoughts, or thinking process to the user. "
                "Do not provide your <think> tag and process"
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
        stream = client.chat.completions.create(
            messages=messages,
            model=model_name,
            stream=True,
        )
        
        complete_response = ""
        for chunk in stream:
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                token = chunk.choices[0].delta.content
                complete_response += token
                token = re.sub(r"<think>.*?</think>", "", token, flags=re.DOTALL).strip()
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"
                    await asyncio.sleep(0.01) 
                    
        complete_response = re.sub(r"<think>.*?</think>", "", complete_response, flags=re.DOTALL).strip()
        
        yield f"data: {json.dumps({'end': True, 'complete_response': complete_response})}\n\n"
        
    except Exception as e:
        print(f"Error in LLM streaming: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

def get_llm_response(context, question, chat_history=""):
    client = Groq(api_key=GROQ_API_KEY)
    model_name = "deepseek-r1-distill-llama-70b"

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
                "Do not provide your <think> tag and process"
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

def get_greeting_response(query, chat_history=""):
    client = Groq(api_key=GROQ_API_KEY)
    model_name = "deepseek-r1-distill-llama-70b"
    
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
                "You are a friendly video chatbot assistant. Respond warmly and briefly to greetings "
                "or questions about your capabilities. Be concise and helpful."
                "Do NOT show your internal reasoning, thoughts, or thinking process to the user. "
                "Do not provide your <think> tag and process"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Chat History:\n{history_str}\n"
                f"User Message: {query}\n"
                f"Please respond to this greeting or introduction:"
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
        print(f"Error in greeting response: {str(e)}")
        return "Hello! How can I help you with your video questions today?"

async def stream_greeting_response(query, chat_history=""):
    client = Groq(api_key=GROQ_API_KEY)
    model_name = "deepseek-r1-distill-llama-70b"
    
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
                "You are a helpful video chatbot. Your job is to answer questions about videos and general topics."
                " Introduce yourself as a video chatbot when asked who you are? Respond warmly and briefly to greetings "
                "or questions about your capabilities. Be concise and helpful."
                "Do NOT show your internal reasoning, thoughts, or thinking process to the user. "
                "Do not provide your <think> tag and process"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Chat History:\n{history_str}\n"
                f"User Message: {query}\n"
                f"Please respond to this greeting or introduction:"
            ),
        },
    ]

    try:
        stream = client.chat.completions.create(
            messages=messages,
            model=model_name,
            stream=True,
        )
        
        complete_response = ""
        for chunk in stream:
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                token = chunk.choices[0].delta.content
                complete_response += token
                token = re.sub(r"<think>.*?</think>", "", token, flags=re.DOTALL).strip()
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"
                    await asyncio.sleep(0.01)  
        complete_response = re.sub(r"<think>.*?</think>", "", complete_response, flags=re.DOTALL).strip()
        
        yield f"data: {json.dumps({'end': True, 'complete_response': complete_response})}\n\n"
        
    except Exception as e:
        print(f"Error in greeting streaming: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

GITHUB_REPO_OWNER = "FawwazRaza"  # Replace with your GitHub username
GITHUB_REPO_NAME = "QueryClip"       # Replace with your repository name
GITHUB_VIDEO_PATH = "data/videos"         # Path to videos folder in your repository

def get_github_video_url(filename):
    """Generate a URL for a video stored in a public GitHub repository"""
    return f"https://raw.githubusercontent.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/main/{GITHUB_VIDEO_PATH}/{filename}"

@app.get("/videos/{video_filename}")
async def get_video(video_filename: str):
    video_path = get_github_video_url(video_filename)
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video {video_filename} not found")
    
    return FileResponse(video_path, media_type="video/mp4")

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "API is running"}

class QueryRequest(BaseModel):
    query: str
    chat_history: list = []
    stream: bool = False

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        route = route_query(request.query)
        print(f"Query routed to: {route}")
        
        if route == "DEFAULT":
            if request.stream:
                return StreamingResponse(
                    stream_greeting_response(request.query, request.chat_history),
                    media_type="text/event-stream"
                )
            else:
                answer = get_greeting_response(request.query, request.chat_history)
                return JSONResponse(content={
                    "answer": answer,
                    "source": None,
                    "chunks": []
                })
        
        elif route == "UNSAFE":
            unsafe_message = "I'm sorry, but I cannot provide information or assistance with that request. Please ask a different question that I can help with."
            if request.stream:
                async def stream_unsafe():
                    for char in unsafe_message:
                        yield f"data: {json.dumps({'token': char})}\n\n"
                        await asyncio.sleep(0.01)
                    yield f"data: {json.dumps({'end': True, 'complete_response': unsafe_message})}\n\n"
                return StreamingResponse(stream_unsafe(), media_type="text/event-stream")
            else:
                return JSONResponse(content={
                    "answer": unsafe_message,
                    "source": None,
                    "chunks": []
                })
        
        else:  
            print("Processing via retriever chain...")
            TOP_K = 3
            retriever_chain = ChromaRetriever().as_retriever(search_kwargs={"k": TOP_K})
            top_chunks = retriever_chain(request.query, k=TOP_K)
            
            print(f"Retrieved {len(top_chunks)} chunks")
            for i, chunk in enumerate(top_chunks):
                print(f"Chunk {i+1} similarity: {chunk.get('similarity', 'N/A')}")
            
            non_empty_chunks = [chunk for chunk in top_chunks if chunk.get("text", "").strip()]
            print(f"Non-empty chunks: {len(non_empty_chunks)}")
            
            if not non_empty_chunks:
                print("No chunks found in database for this query")
                not_found_message = "Not found in the dataset."
                
                if request.stream:
                    async def stream_not_found():
                        for char in not_found_message:
                            yield f"data: {json.dumps({'token': char})}\n\n"
                            await asyncio.sleep(0.01)
                        yield f"data: {json.dumps({'end': True, 'complete_response': not_found_message})}\n\n"
                    return StreamingResponse(stream_not_found(), media_type="text/event-stream")
                else:
                    return JSONResponse(content={
                        "answer": not_found_message,
                        "source": None,
                        "chunks": []
                    })
            
            top_chunk = max(non_empty_chunks, key=lambda c: c.get("similarity", 0))
            print(f"Top chunk from: {top_chunk.get('file_name', 'unknown')}")
            
            combined_context = "\n\n".join([chunk["text"] for chunk in non_empty_chunks])
            context = (
                f"{combined_context}\n\n"
                f"Most relevant source: (File: {top_chunk['file_name']}, "
                f"Start: {top_chunk['start_time']}s, End: {top_chunk['end_time']}s)"
            )
            
            print("Sending context to LLM for answer generation...")
            
            if request.stream:
                source_info = {
                    "file_name": top_chunk["file_name"],
                    "start_time": top_chunk["start_time"],
                    "end_time": top_chunk["end_time"]
                }
                
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
                
                async def stream_with_metadata():
                    yield f"data: {json.dumps({'metadata': {
                        'source': source_info,
                        'chunks': chunk_payload
                    }})}\n\n"
                    
                    async for token in stream_llm_response(context, request.query, request.chat_history):
                        yield token
                
                return StreamingResponse(stream_with_metadata(), media_type="text/event-stream")
            else:
                answer = get_llm_response(context, request.query, request.chat_history)
                
                print(f"Generated answer: {answer[:100]}...")
                
                if "Not found in the dataset" in answer:
                    print("Answer not found in context")
                    return JSONResponse(content={
                        "answer": answer,
                        "source": None,
                        "chunks": []
                    })
                else:
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
                    
                    source_info = {
                        "file_name": top_chunk["file_name"],
                        "start_time": top_chunk["start_time"],
                        "end_time": top_chunk["end_time"]
                    }
                    
                    print(f"Returning answer with source from {source_info['file_name']}")
                    return JSONResponse(content={
                        "answer": answer,
                        "source": source_info,
                        "chunks": chunk_payload
                    })
            
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    pass

# Function to start Ngrok tunnel
def start_ngrok():
    # Start ngrok tunnel to expose the API
    public_url = ngrok.connect(8000).public_url
    print(f"ngrok tunnel active at: {public_url}")
    with open("ngrok_url.txt", "w") as f:
        f.write(public_url)
    return public_url

if __name__ == "__main__":
    # Start Ngrok when this file is run directly
    ngrok_url = start_ngrok()
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)