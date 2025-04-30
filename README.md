# QueryClip - AI Chatbot with Video Understanding and RAG

![Flow Diagram](https://github.com/FawwazRaza/QueryClip/blob/main/Flow%20diagrams/complete%20flow.png)

Welcome to **QueryClip** – an AI-powered chatbot that understands your video content. This project uses cutting-edge technology like embeddings, chunking, and Retrieval-Augmented Generation (RAG) to give smart, accurate answers based on your own data.

This guide will help you set up the chatbot step-by-step.

---

##  What is QueryClip?

QueryClip is a smart chatbot that can understand and answer questions based on your uploaded videos. It uses:

- A **frontend** (simple user interface) where users ask questions.
- A **backend** that processes questions and talks to a language model (LLM).
- A **vector database** that stores knowledge from your videos.
- A full **pipeline** that extracts text from videos, breaks it into parts (chunks), finds important meanings (embeddings), and helps the chatbot answer using this data.

---

##  How to Use QueryClip

### 1. Clone the Project

Open your terminal and run:

```bash
git clone https://github.com/FawwazRaza/QueryClip.git
cd QueryClip

### 2. Add Your Videos
Put your .mp4 video files into the following folder:

```bash
QueryClip/data/videos/
