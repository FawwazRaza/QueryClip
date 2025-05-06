import streamlit as st
import requests
import os
import time
import json
from typing import Iterator
import sseclient

# VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/videos")
# os.makedirs(VIDEO_DIR, exist_ok=True)


# def display_video_from_backend(video_filename, start_time=0):
#     video_url = f"{NGROK_URL}/videos/{video_filename}"
    
#     # Check if video exists on backend
#     try:
#         # Using HEAD request to check if video exists without downloading
#         response = requests.head(video_url, timeout=5)
#         if response.status_code == 200:
#             # Use st.video with the URL
#             st.video(video_url, start_time=int(float(start_time)))
#         else:
#             st.warning(f"Video {video_filename} not found on backend server.")
#     except Exception as e:
#         st.error(f"Error accessing video: {str(e)}")


def get_github_video_url(repo_owner, repo_name, path_to_video):
    """Generate a URL for a video stored in a public GitHub repo"""
    return f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/main/{path_to_video}"



NGROK_URL = st.secrets.get("NGROK_URL", os.getenv("NGROK_URL", ""))

# If running locally and we have a saved URL, use that
if not NGROK_URL and os.path.exists("ngrok_url.txt"):
    with open("ngrok_url.txt", "r") as f:
        NGROK_URL = f.read().strip()

if not NGROK_URL:
    st.error("No Ngrok URL found! The backend server is not accessible.")
    st.info("You need to run ngrok_backend.py first and set the URL in Streamlit secrets.")
    st.stop()

API_URL = f"{NGROK_URL}/query"
HEALTH_CHECK_URL = f"{NGROK_URL}/"

def is_api_available():
    try:
        response = requests.get(HEALTH_CHECK_URL, timeout=5)
        return response.status_code == 200
    except:
        return False

def handle_api_error(message="An error occurred"):
    st.error(f" {message}")
    st.info("If this is your first time running the app, please check:")
    st.markdown("""
    1. Is the backend server running?
    2. Have you set your GROQ API key in the `.env` file?
    3. Are there any videos in the `../data/videos` directory?
    """)
    st.info("Run the following command in your terminal to start all services:")
    st.code("python run_app.py")

st.set_page_config(
    page_title="RAG Video Chatbot",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.image("https://www.groq.com/images/logo.svg", width=100)
    st.title("RAG Video Chatbot")
    
    if not is_api_available():
        st.error(" Backend API is not available")
    else:
        st.success("Connected to backend API")
    
    if os.path.exists(VIDEO_DIR):
        videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        st.write(f" Found {len(videos)} videos in library")
        if videos:
            with st.expander("Video Library"):
                for video in videos:
                    st.write(f"- {video}")
        else:
            st.warning("No videos found. Add videos to the ../data/videos directory.")
    else:
        st.error(f"Video directory not found: {VIDEO_DIR}")
        if st.button("Create Video Directory"):
            os.makedirs(VIDEO_DIR, exist_ok=True)
            st.success(f"Created directory: {VIDEO_DIR}")
            st.rerun()
    
    with st.expander("Help & Information"):
        st.markdown("""
        ### How to use this chatbot:

        - **Ask about videos**: Query specific information from your video library
        - **General questions**: The bot can also answer general questions
        - **Commands**: Try typing "help" or "clear" to see special commands

        ### Adding videos:
        Place your video files in the `../data/videos` directory
        """)
    
    streaming_enabled = st.checkbox("Enable streaming", value=True, help="Show tokens as they're generated")
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

st.title(" Video Knowledge Chatbot")
st.markdown("Ask me anything about your videos or any general questions!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(message["content"])
            
            if "source" in message and message["source"]:
                src = message["source"]
                video_path = os.path.join(VIDEO_DIR, src['file_name'])
                if os.path.exists(video_path):
                    st.markdown(
                        f"**Source:** `{src['file_name']}` | Time: `{src['start_time']}s - {src['end_time']}s`"
                    )
                    try:
                        video_url = get_github_video_url('FawwazRaza', 'QueryClip', f"data/videos/{src['file_name']}")
                        st.video(video_url, start_time=int(float(src['start_time'])))
                    except Exception as e:
                        st.error(f"Error playing video: {str(e)}")
                else:
                    st.warning(f"Video file `{src['file_name']}` not found in `{VIDEO_DIR}`.")
                
                if "chunks" in message and message["chunks"]:
                    with st.expander(" View Related Video Contexts"):
                        for idx, chunk in enumerate(message["chunks"], 1):
                            similarity = chunk.get('similarity', 0)
                            if similarity > 0.8:
                                sim_color = "green"
                            elif similarity > 0.5:
                                sim_color = "blue"
                            else:
                                sim_color = "red"
                                
                            st.markdown(f"##### Chunk {idx} - Relevance: :{sim_color}[{similarity:.2f}]")
                            st.markdown(f"*Source: {chunk['file_name']}, Time: {chunk['start_time']}s - {chunk['end_time']}s*")
                            st.markdown(f"```\n{chunk['text']}\n```")
                            st.markdown("---")

def process_special_commands(query):
    query_lower = query.lower().strip()
    
    if query_lower == "clear":
        st.session_state.chat_history = []
        st.rerun()
        return True
    
    if query_lower in ["help", "commands"]:
        help_message = {
            "role": "assistant",
            "content": """
            ### Available Commands:
            - **clear**: Clear the chat history
            - **help**: Display this help message
            
            ### Example Questions:
            - "Tell me about the content in [video name]"
            - "What are the key points discussed in the videos?"
            - "Can you summarize the information about [topic]?"
            """
        }
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append(help_message)
        st.rerun()
        return True
    
    return False

def process_stream_manually(response):
    """Process the SSE stream manually without using sseclient."""
    buffer = ""
    source_info = None
    chunks_info = None
    complete_response = ""
    
    try:
        for line in response.iter_lines():
            if not line:
                continue
                
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data_str = line[6:]  
                try:
                    data = json.loads(data_str)
                    
                    if 'metadata' in data:
                        metadata = data['metadata']
                        source_info = metadata.get('source')
                        chunks_info = metadata.get('chunks')
                        continue
                    
                    if 'token' in data:
                        token = data['token']
                        if '<think>' in token.lower():  
                            continue
                        complete_response += token
                        yield {'type': 'token', 'content': token}

                    
                    if 'end' in data and data['end']:
                        if 'complete_response' in data:
                            complete_response = data['complete_response']
                        yield {'type': 'complete', 'content': complete_response, 'source': source_info, 'chunks': chunks_info}
                    
                    if 'error' in data:
                        yield {'type': 'error', 'content': f"Error: {data['error']}"}
                        
                except json.JSONDecodeError:
                    yield {'type': 'error', 'content': f"Error decoding response: {data_str}"}
                    
    except Exception as e:
        yield {'type': 'error', 'content': f"Error processing stream: {str(e)}"}

query = st.chat_input("Type your message here...")

if query:
    if process_special_commands(query):
        pass  
    else:
        with st.chat_message("user", avatar="👤"):
            st.write(query)
        
        st.session_state.chat_history.append({"role": "user", "content": query})

        if not is_api_available():
            handle_api_error("Backend API is not available")
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": " I'm having trouble connecting to the backend. Please check if the server is running."
            })
            st.rerun()

        with st.chat_message("assistant", avatar="🤖") as message_container:
            if streaming_enabled:
                try:
                    response_placeholder = st.empty()
                    collected_tokens = ""
                    source_data = None
                    chunks_data = None
                    
                    with st.spinner("Connecting..."):
                        response = requests.post(
                            API_URL,
                            json={
                                "query": query,
                                "chat_history": st.session_state.chat_history[:-1],
                                "stream": True
                            },
                            stream=True,
                            timeout=60,
                            headers={'Accept': 'text/event-stream'} 
                        )
                    
                    if response.status_code == 200:
                        stream_generator = process_stream_manually(response)
                        for message in stream_generator:
                            if message['type'] == 'token':
                                collected_tokens += message['content']
                                response_placeholder.markdown(collected_tokens + "▌")  
                            
                            elif message['type'] == 'complete':
                                final_response = message['content']
                                source_data = message.get('source')
                                chunks_data = message.get('chunks')
                                response_placeholder.markdown(final_response)  
                                
                                assistant_message = {
                                    "role": "assistant", 
                                    "content": final_response
                                }
                                
                                if source_data and "Not found in the dataset" not in final_response:
                                    assistant_message["source"] = source_data
                                    if chunks_data:
                                        assistant_message["chunks"] = chunks_data
                                
                                st.session_state.chat_history.append(assistant_message)
                                
                                if source_data and "Not found in the dataset" not in final_response:
                                    src = source_data
                                    video_path = os.path.join(VIDEO_DIR, src['file_name'])
                                    if os.path.exists(video_path):
                                        st.markdown(
                                            f"**Source:** `{src['file_name']}` | Time: `{src['start_time']}s - {src['end_time']}s`"
                                        )
                                        try:
                                            video_url = get_github_video_url('FawwazRaza', 'QueryClip', f"data/videos/{src['file_name']}")
                                            st.video(video_url, start_time=int(float(src['start_time'])))
                                        except Exception as e:
                                            st.error(f"Error playing video: {str(e)}")
                                    else:
                                        st.warning(f"Video file `{src['file_name']}` not found in `{VIDEO_DIR}`.")
                                    
                                    if chunks_data:
                                        with st.expander(" View Related Video Contexts"):
                                            for idx, chunk in enumerate(chunks_data, 1):
                                                similarity = chunk.get('similarity', 0)
                                                if similarity > 0.8:
                                                    sim_color = "green"
                                                elif similarity > 0.5:
                                                    sim_color = "blue"
                                                else:
                                                    sim_color = "red"
                                                    
                                                st.markdown(f"##### Chunk {idx} - Relevance: :{sim_color}[{similarity:.2f}]")
                                                st.markdown(f"*Source: {chunk['file_name']}, Time: {chunk['start_time']}s - {chunk['end_time']}s*")
                                                st.markdown(f"```\n{chunk['text']}\n```")
                                                st.markdown("---")
                            
                            elif message['type'] == 'error':
                                st.error(message['content'])
                                st.session_state.chat_history.append({
                                    "role": "assistant", 
                                    "content": message['content']
                                })
                    
                    else:
                        error_msg = f"Server error: {response.status_code}"
                        try:
                            error_details = response.json()
                            if "detail" in error_details:
                                error_msg += f" - {error_details['detail']}"
                        except:
                            pass
                        
                        st.error(error_msg)
                        st.info("Please check the backend server logs for more information.")
                        
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": f" {error_msg}"
                        })
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": f" Error: {str(e)}"
                    })
            
            else:
                try:
                    with st.spinner("Getting answer..."):
                        response = requests.post(
                            API_URL,
                            json={
                                "query": query,
                                "chat_history": st.session_state.chat_history[:-1],
                                "stream": False
                            },
                            timeout=60
                        )
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        answer = response_data.get("answer", "No answer received")
                        source_data = response_data.get("source")
                        chunks_data = response_data.get("chunks", [])
                        
                        st.write(answer)
                        
                        assistant_message = {
                            "role": "assistant", 
                            "content": answer
                        }
                        
                        if source_data and "Not found in the dataset" not in answer:
                            assistant_message["source"] = source_data
                            if chunks_data:
                                assistant_message["chunks"] = chunks_data
                        
                        st.session_state.chat_history.append(assistant_message)
                        
                        if source_data and "Not found in the dataset" not in answer:
                            src = source_data
                            video_path = os.path.join(VIDEO_DIR, src['file_name'])
                            if os.path.exists(video_path):
                                st.markdown(
                                    f"**Source:** `{src['file_name']}` | Time: `{src['start_time']}s - {src['end_time']}s`"
                                )
                                try:
                                    video_url = get_github_video_url('FawwazRaza', 'QueryClip', f"data/videos/{src['file_name']}")
                                    st.video(video_url, start_time=int(float(src['start_time'])))
                                except Exception as e:
                                    st.error(f"Error playing video: {str(e)}")
                            else:
                                st.warning(f"Video file `{src['file_name']}` not found in `{VIDEO_DIR}`.")
                            
                            if chunks_data:
                                with st.expander(" View Related Video Contexts"):
                                    for idx, chunk in enumerate(chunks_data, 1):
                                        similarity = chunk.get('similarity', 0)
                                        if similarity > 0.8:
                                            sim_color = "green"
                                        elif similarity > 0.5:
                                            sim_color = "blue"
                                        else:
                                            sim_color = "red"
                                            
                                        st.markdown(f"##### Chunk {idx} - Relevance: :{sim_color}[{similarity:.2f}]")
                                        st.markdown(f"*Source: {chunk['file_name']}, Time: {chunk['start_time']}s - {chunk['end_time']}s*")
                                        st.markdown(f"```\n{chunk['text']}\n```")
                                        st.markdown("---")
                    
                    else:
                        error_msg = f"Server error: {response.status_code}"
                        try:
                            error_details = response.json()
                            if "detail" in error_details:
                                error_msg += f" - {error_details['detail']}"
                        except:
                            pass
                        
                        st.error(error_msg)
                        st.info("Please check the backend server logs for more information.")
                        
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": f"  {error_msg}"
                        })

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": f"  Error: {str(e)}"
                    })

if not st.session_state.chat_history:
    with st.chat_message("assistant", avatar="🤖"):
        st.write("""
         Welcome to the RAG Video Chatbot!
        
        I can help you find information from your video library. Try asking me questions about any video content you've added to the system.
        
        Type 'help' to see available commands.
        """)