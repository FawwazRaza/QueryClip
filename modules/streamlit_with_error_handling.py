

import streamlit as st
import requests
import os
import time

VIDEO_DIR = "../data/videos"
API_URL = "http://localhost:8000/query"

# Function to check if the API is available
def is_api_available():
    try:
        response = requests.get(API_URL.replace("/query", "/"))
        return response.status_code == 200
    except:
        return False

st.set_page_config(page_title="RAG Video Chatbot", page_icon=None)
st.title("RAG Video Chatbot")
st.markdown("Ask me anything! I can answer general questions or provide information from your video library.")

# Check if API is running
api_status = st.empty()
if not is_api_available():
    api_status.error("⚠️ Backend API is not available. Please start the FastAPI server first.")
    st.info("Run the following command in your terminal:")
    st.code("python -m uvicorn fastapi_backend_updated:app --reload --host 0.0.0.0 --port 8000")
else:
    api_status.success("✅ Connected to backend API")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])
            # If there's a source, display video
            if "source" in message and message["source"]:
                src = message["source"]
                video_path = os.path.join(VIDEO_DIR, src['file_name'])
                if os.path.exists(video_path):
                    st.markdown(
                        f"**Source:** `{src['file_name']}` | Start: `{src['start_time']}s` | End: `{src['end_time']}s`"
                    )
                    st.video(video_path, start_time=int(src['start_time']))

# Get user input using chat_input instead of text_input for better chat experience
query = st.chat_input("Type your message here...")

if query:
    # Display user message
    with st.chat_message("user"):
        st.write(query)
    
    # Add to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                API_URL,
                json={
                    "query": query,
                    "chat_history": st.session_state.chat_history
                },
                timeout=30  # Set a reasonable timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.write(data['answer'])
                    
                    # Only display source information if it exists and the answer is not "Not found in the dataset"
                    if "source" in data and data["source"] and "Not found in the dataset" not in data['answer']:
                        src = data["source"]
                        video_path = os.path.join(VIDEO_DIR, src['file_name'])
                        if os.path.exists(video_path):
                            st.markdown(
                                f"**Source:** `{src['file_name']}` | Start: `{src['start_time']}s` | End: `{src['end_time']}s`"
                            )
                            st.video(video_path, start_time=int(src['start_time']))
                        else:
                            st.warning(f"Video file `{src['file_name']}` not found in `{VIDEO_DIR}`.")
                        
                        # Show expandable section with relevant chunks only if they exist
                        if "chunks" in data and data["chunks"]:
                            with st.expander("View relevant video context"):
                                for idx, chunk in enumerate(data["chunks"], 1):
                                    st.markdown(f"**Chunk {idx}** (Similarity: {chunk.get('similarity', 'N/A'):.3f})")
                                    st.markdown(f"*File: {chunk['file_name']}, Time: {chunk['start_time']}s - {chunk['end_time']}s*")
                                    st.markdown(chunk["text"])
                                    st.markdown("---")
                
                # Add assistant response to chat history with source if available
                assistant_message = {
                    "role": "assistant", 
                    "content": data['answer']
                }
                if "source" in data and data["source"] and "Not found in the dataset" not in data['answer']:
                    assistant_message["source"] = data["source"]
                
                st.session_state.chat_history.append(assistant_message)
            else:
                with st.chat_message("assistant"):
                    st.error(f"Server error: {response.status_code}")
                    st.info("Please check the backend server logs for more information.")
                st.session_state.chat_history.append({"role": "assistant", "content": f"Server error: {response.status_code}. Please try again."})
        
        except requests.exceptions.ConnectionError:
            with st.chat_message("assistant"):
                st.error("Cannot connect to the backend server. Please make sure it's running.")
                st.info("Run the following command in your terminal:")
                st.code("python -m uvicorn fastapi_backend_updated:app --reload --host 0.0.0.0 --port 8000")
            st.session_state.chat_history.append({"role": "assistant", "content": "Connection error. Cannot reach the backend server."})
        
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"An error occurred: {str(e)}")
            st.session_state.chat_history.append({"role": "assistant", "content": f"An error occurred: {str(e)}"})