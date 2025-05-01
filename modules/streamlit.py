import streamlit as st
import requests
import os

VIDEO_DIR = "../data/videos"
API_URL = "http://localhost:8000/query"

st.set_page_config(page_title="RAG Video Chatbot", page_icon=None)
st.title("RAG Video Chatbot")
st.markdown("Ask a question about your videos. The answer and supporting context will be shown below.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# for message in st.session_state.chat_history:
#     if message["role"] == "user":
#         st.chat_message("user").write(message["content"])
#     else:
#         st.chat_message("assistant").write(message["content"])

query = st.text_input("Enter your question:", "")

if st.button("Get Answer") and query.strip():
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.spinner("Processing your request..."):
        response = requests.post(
            API_URL,
            json={
                "query": query,
                "chat_history": st.session_state.chat_history
            }
        )

    if response.status_code == 200:
        data = response.json()
        st.session_state.chat_history.append({"role": "assistant", "content": data['answer']})

        st.markdown("### Answer")
        st.markdown(data['answer'])
        if "source" in data and data["source"].get("file_name"):
            src = data["source"]
            video_path = os.path.join(VIDEO_DIR, src['file_name'])
            if os.path.exists(video_path):
                st.markdown(
                    f"**Source:** `{src['file_name']}` | Start: `{src['start_time']}s` | End: `{src['end_time']}s`"
                )
                st.video(video_path, start_time=int(src['start_time']))
            else:
                st.warning(f"Video file `{src['file_name']}` not found in `{VIDEO_DIR}`.")
        st.markdown("---")
        st.markdown("### Top 3 Relevant Chunks")
        for idx, chunk in enumerate(data["chunks"], 1):
            with st.expander(
                f"Chunk {idx} (File: {chunk['file_name']}, Start: {chunk['start_time']}s, End: {chunk['end_time']}s, Similarity: {chunk.get('similarity', 'N/A'):.3f})",
                expanded=False
            ):
                st.write(chunk["text"])


    else:
        st.error("Could not retrieve a response from the backend.")