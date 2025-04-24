# Salesforce Earnings RAG

This repository implements a Retrieval-Augmented Generation (RAG) system for Salesforce earnings call transcripts. It provides Question & Answer (Q&A) and summarization capabilities via a simple Conversational UI.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Setup](#setup)
  - [1. Clone & Environment](#1-clone--environment)
  - [2. Install Dependencies](#2-install-dependencies)
  - [3. Environment Variables](#3-environment-variables)
  - [4. Data Preparation](#data-preparation)
  - [5. Indexing Chunks](#indexing-chunks)
  - [6. Running the API](#running-the-api)
  - [7. Running the UI](#running-the-ui)

---

## Features

- **Question Answering**: Concise answers with inline citations (`[1]`, `[2]`, etc.)
- **Summarization**: Topic- or trend-based summaries
- **Inline Citations**: Maps model citations back to chunk IDs and snippets
- **Collapsible Snippets**: Streamlit expanders link to original PDF pages
- **Interactive Filters**: Can filter for infromation from specific quarters or speakers
- **Batch Embeddings**: Uses parallel workers to process transcripts faster

---

## Architecture

1. **Data Ingestion & Chunking**: Convert PDFs to text, clean, and split into overlapping chunks
2. **Embeddings & Vector Store**: Compute embeddings with OpenAI (`text-embedding-ada-002`) and upsert into Pinecone
3. **Retrieval Layer**: Embed queries and k-NN search against Pinecone with metadata filters
4. **LLM Generation**: Use GPT-4 to answer or summarize with retrieved snippets and inline citations
5. **Conversational UI**: Streamlit app to drive Q&A and summarization

---

## Prerequisites

- Python 3.8+
- Dependencies in requirements.txt

---

## Project Structure

```bash
RAGSystemDesign/
├── .env.example            # Example environment variables
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/                # Source PDF & extracted .txt
│   └── chunks/             # Chunked text files
│
├── scripts/
│   ├── convert_pdfs.py     # PDF → text extraction
│   └── prepare_chunks.py   # Text cleaning & chunking
│
├── embeddings/
│   └── indexer.py          # Compute embeddings & upsert to Pinecone
│
├── retriever/
│   └── retriever.py        # Vector query
│
├── api/
│   └── server.py           # FastAPI app (QA & summarization)
│
├── ui/
│   └── app.py              # Streamlit front-end
│
└── tests/                  # Unit & integration tests
    ├── test_retriever.py
    ├── test_indexer.py
    └── test_api.py
```

---

## Setup

### 1. Clone & Environment

```bash
git clone https://github.com/your-org/salesforce-rag.git
cd RAGSystemDesign
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment Variables

Copy the example and fill in your keys:

```bash
cp .env.example .env
# Edit .env with your OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX
```

---

## 4. Data Preparation

Convert PDFs to text and chunk:

```bash
python scripts/convert_pdfs.py    # Extract text files in data/raw/
python scripts/prepare_chunks.py  # Create overlapping chunks in data/chunks/
```

---

## 5. Indexing Chunks

Upsert all chunks into Pinecone (batchable):

```bash
python embeddings/indexer.py
```

---

## 6. Running the API

Start the FastAPI server (port 8000):

```bash
uvicorn api.server:app --reload --port 8000
```

### 7. Endpoints

- `POST /qa` — Q&A with inline citations
- `POST /summarize` — Summarize key points

---

## 8. Running the UI

Launch the Streamlit app (port 8501):

```bash
streamlit run ui/app.py
```

Open your browser to `http://localhost:8501` to interact.

---
