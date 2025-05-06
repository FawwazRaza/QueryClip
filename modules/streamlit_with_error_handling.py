import streamlit as st
import requests
import os
import time
import json

VIDEO_DIR = "../data/videos"
API_URL = "http://localhost:8000/query"
HEALTH_CHECK_URL = "http://localhost:8000/"

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
    st.markdown("##### Powered by Groq LLM API")
    
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
                        st.video(video_path, start_time=int(float(src['start_time'])))
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
                "content": "⚠️ I'm having trouble connecting to the backend. Please check if the server is running."
            })
            st.rerun()

        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "query": query,
                        "chat_history": st.session_state.chat_history[:-1]  
                    },
                    timeout=30  
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    assistant_message = {
                        "role": "assistant", 
                        "content": data['answer']
                    }
                    
                    if "source" in data and data["source"] and "Not found in the dataset" not in data['answer']:
                        assistant_message["source"] = data["source"]
                        if "chunks" in data and data["chunks"]:
                            assistant_message["chunks"] = data["chunks"]
                    
                    st.session_state.chat_history.append(assistant_message)
                    
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(data['answer'])
                        
                        if "source" in data and data["source"] and "Not found in the dataset" not in data['answer']:
                            src = data["source"]
                            video_path = os.path.join(VIDEO_DIR, src['file_name'])
                            if os.path.exists(video_path):
                                st.markdown(
                                    f"**Source:** `{src['file_name']}` | Time: `{src['start_time']}s - {src['end_time']}s`"
                                )
                                try:
                                    st.video(video_path, start_time=int(float(src['start_time'])))
                                except Exception as e:
                                    st.error(f"Error playing video: {str(e)}")
                            else:
                                st.warning(f"Video file `{src['file_name']}` not found in `{VIDEO_DIR}`.")
                            
                            if "chunks" in data and data["chunks"]:
                                with st.expander(" View Related Video Contexts"):
                                    for idx, chunk in enumerate(data["chunks"], 1):
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
                    
                    with st.chat_message("assistant", avatar="🤖"):
                        st.error(error_msg)
                        st.info("Please check the backend server logs for more information.")
                    
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": f" {error_msg}. Please try again."
                    })
            
            except requests.exceptions.ConnectionError:
                handle_api_error("Cannot connect to the backend server")
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": " Connection error. Cannot reach the backend server."
                })
            
            except requests.exceptions.Timeout:
                handle_api_error("Request timed out. The server might be overloaded.")
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": " Request timed out. Please try again or ask a simpler question."
                })
            
            except Exception as e:
                with st.chat_message("assistant", avatar="🤖"):
                    st.error(f"An unexpected error occurred: {str(e)}")
                
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": f" An unexpected error occurred: {str(e)}"
                })
                
        st.rerun()

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 0.8em;">
        RAG Video Chatbot | Powered by <a href="https://groq.com" target="_blank">Groq</a> | 
        Built with <a href="https://streamlit.io" target="_blank">Streamlit</a>
    </div>
    """,
    unsafe_allow_html=True
)