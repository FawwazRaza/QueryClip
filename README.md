# QueryClip - AI Chatbot with Video Understanding and RAG

<p align="center">
  <img src="https://github.com/FawwazRaza/QueryClip/blob/main/Flow%20diagrams/complete%20flow.png" alt="QueryClip Architecture" width="800">
</p>

## Overview

QueryClip is an AI-powered chatbot that understands and answers questions about your video content. It uses cutting-edge technology including:

- **Speech-to-Text Transcription**: Extracts spoken content from videos
- **Text Chunking**: Breaks transcripts into manageable pieces
- **Embeddings**: Converts text into meaningful vector representations
- **Vector Database**: Stores and retrieves knowledge efficiently
- **Retrieval-Augmented Generation (RAG)**: Enhances LLM responses with relevant context

With QueryClip, you can upload videos and immediately start asking questions about their content. The system will provide accurate, contextual answers based on what was actually said in the videos.

## Features

- **Video Content Understanding**: Automatically transcribes and processes video content
- **Natural Language Querying**: Ask questions in plain language
- **Contextual Responses**: Get accurate answers based on your video content
- **Simple User Interface**: Easy-to-use frontend for interacting with the chatbot
- **Modular Architecture**: Well-structured codebase for easy customization

## Requirements

- Python 3.8+
- Internet connection (for API access)
- [Groq](https://groq.com/) API key (free tier available)
- Git

## Quick Start

### 1. Clone the Project

```bash
git clone https://github.com/FawwazRaza/QueryClip.git
cd QueryClip
```

### 2. Add Your Videos

Place your `.mp4` video files in the videos directory:

```bash
Add your video files to data/videos/
```

### 3. Set Up Environment

Create a virtual environment and install dependencies:

```bash
conda create -n myenv
conda activate myenv
# or
conda activate 
pip install -r requirements.txt
```

### 4. Add Your API Key

Sign up for a free [Groq API key](https://console.groq.com/signup).

Write in a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

### 5. Process Videos and Start the Chatbot

Run each module in sequence:

```bash
cd modules

# Step 1: Transcribe videos to text
python transcriber.py

# Step 2: Chunk the transcribed text
python chunker.py

# Step 3: Generate embeddings
python embedder.py

# Step 4: Store embeddings in vector database
python vector_store.py

# Step 5: Run the QA engine backend
python qa_engine.py
```

### 6. Launch the User Interface

Open two terminal windows:

**Terminal 1 - Run the FastAPI backend:**
```bash
cd modules
uvicorn fastapi_backend:app --reload --port 8000
```

**Terminal 2 - Run the Streamlit frontend:**
```bash
cd modules
streamlit run streamlit.py
```

### 7. Start Chatting!

Open your browser to the URL shown by Streamlit (typically http://localhost:8501) and start asking questions about your videos!

## Project Structure

```
QueryClip/
├── data/
│   ├── videos/           # Where you place your video files
│   ├── transcripts/      # Generated text from videos
│   ├── chunks/           # Segmented text chunks
│   └── embeddings/       # Vector representations
├── modules/
│   ├── transcriber.py    # Converts video to text
│   ├── chunker.py        # Breaks the text into chunks
│   ├── embedder.py       # Creates embeddings
│   ├── vector_store.py   # Saves embeddings in vector DB
│   ├── qa_engine.py      # Handles RAG operations
│   ├── fastapi_backend.py # API endpoints
│   └── streamlit.py      # User interface
├── Flow diagrams/        # Architecture diagrams
├── .env                  # Environment variables (API keys)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Advanced Usage

### Customizing Chunk Size

You can modify the chunk size in `chunker.py` to balance between context window limitations and maintaining coherence.

### Using Different LLM Providers

While the default is set to use Groq, you can modify `qa_engine.py` to use other providers like OpenAI, Anthropic, or local models.

### Batch Processing

For processing multiple videos at once, you can run the transcription and processing steps in batch mode.

## Acknowledgements

- [Groq](https://groq.com/) for providing fast LLM inference
- [Sentence Transformers](https://www.sbert.net/) for embedding generation
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Whisper](https://github.com/openai/whisper) for audio transcription
- [FastAPI](https://fastapi.tiangolo.com/) and [Streamlit](https://streamlit.io/) for the web interface
