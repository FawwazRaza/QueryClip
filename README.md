# QueryClip

A YouTube transcript RAG chatbot. Paste a YouTube video or playlist URL, extract the transcript automatically, and ask questions grounded strictly in the video content. Powered by Groq LLM, ChromaDB, and Streamlit.

## Architecture

```
User (Streamlit) → FastAPI backend (ngrok tunnel)
                        ↓
             [Process YouTube URL]
             yt-dlp       → extract video list
             youtube-transcript-api → fetch captions
             sentence-transformers  → embed chunks
             ChromaDB               → store vectors
                        ↓
             [Query]
             embed query → ChromaDB search → top-4 chunks
             Groq LLM    → answer grounded in transcript
```

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/FawwazRaza/QueryClip.git
cd QueryClip
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | From https://console.groq.com |
| `NGROK_AUTH_TOKEN` | From https://dashboard.ngrok.com/get-started/your-authtoken |

### 3. Run the backend

```bash
python ngrok_backend.py
```

This starts FastAPI on port 8000 and opens a tunnel at `https://great-repeatedly-alien.ngrok-free.app`.

### 4. Run the Streamlit app

In a separate terminal:

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Usage

1. In the sidebar, paste a YouTube video or playlist URL.
2. Click **Process** and wait for indexing (a few seconds per video).
3. Ask questions in the chat. Answers are drawn exclusively from the transcript.

### Special commands

| Command | Action |
|---|---|
| `/clear` | Clear chat history |
| `/help` | Show usage instructions |

## Streamlit Cloud deployment

1. Push the repo to GitHub (`.env` is gitignored — never committed).
2. Deploy at https://share.streamlit.io.
3. In Streamlit Cloud secrets, add:
   ```toml
   BACKEND_URL = "https://great-repeatedly-alien.ngrok-free.app"
   ```
4. Keep `python ngrok_backend.py` running locally to serve the backend.

## Project structure

```
QueryClip/
├── app.py                  Streamlit frontend
├── ngrok_backend.py        Backend launcher with ngrok tunnel
├── requirements.txt
├── .env.example
├── .gitignore
├── backend/
│   ├── transcript_loader.py   yt-dlp + youtube-transcript-api
│   ├── vector_store.py        ChromaDB CRUD
│   ├── rag_chain.py           Groq LLM + RAG pipeline
│   └── fastapi_app.py         REST API endpoints
└── data/
    └── chroma_db/             ChromaDB persistent storage (gitignored)
```

## Troubleshooting

**"Backend is offline"** — Run `python ngrok_backend.py` and click Check Connection in the sidebar.

**"No transcripts could be fetched"** — The video may have captions disabled. Try a different video.

**Groq rate limit errors** — The free tier has per-minute token limits. Wait a moment and retry.

**ngrok domain not connecting** — Ensure `NGROK_AUTH_TOKEN` is set correctly in `.env`. Static domains require an authenticated account.

