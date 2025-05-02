# Salesforce Earnings RAG

Conversational Retrieval‑Augmented Generation for Salesforce earnings‑call transcripts.
Ask questions, get cited answers or summaries, and explore source snippets - all from a single chat UI.

---

## Table of Contents

* [Features](#features)
* [Architecture](#architecture)
* [Prerequisites](#prerequisites)
* [Project Structure](#project-structure)
* [Setup](#setup)

  * [1. Clone & Environment](#1-clone--environment)
  * [2. Install Dependencies](#2-install-dependencies)
  * [3. Environment Variables](#3-environment-variables)
  * [4. Build the Index](#4-build-the-index)
  * [5. Run the API](#5-run-the-api)
  * [6. Run the UI](#6-run-the-ui)
* [Usage Examples](#usage-examples)

---

## Features

* **Conversational QA** – multi‑turn chat with memory
* **Summarization** – concise topic or trend summaries
* **Inline Citations** – facts cited like `[1]`; UI shows up to 4 snippets
* **Meta Queries** – programmatic answers (doc count, page count)
* **Polite Small‑Talk** – graceful replies to “Thanks”, “Hi”, etc.
* **Clear Chat** – one‑click history reset

---

## Architecture

1. **Ingest & Chunk** – PDFs → overlapping 1 000‑token chunks
2. **Embeddings + Store** – OpenAI embeddings upserted to Pinecone
3. **Retriever** – k‑NN search (k = 8) with metadata
4. **LLM Chain** – GPT‑4o‑mini combines snippets, cites sources
5. **FastAPI** – `/chat` endpoint handles memory + meta + RAG
6. **Streamlit UI** – chat interface with collapsible source snippets

---

## Prerequisites

* Python 3.9+
* Keys for **OpenAI** and **Pinecone**

---

## Project Structure

```bash
RAGSystemDesign/
├── api/
│   └── server.py             # FastAPI back‑end
├── ui/
│   └── app.py                # Streamlit chat front‑end
├── scripts/
│   └── build_index.py        # Ingest → chunk → embed → upsert
├── data/
│   └── raw/                  # Store earnings call PDFs
├── archive/
│   └── *                     # Deprecated scripts from old design
├── requirements.txt
├── Architecture_Diagram.pdf  # Visual representation of system design
├── .env.example
├── docker-compose.yml        # Can be used in future for containerization + deployment
└── .gitignore
```

---

## Setup

### 1. Clone & Environment

```bash
git clone https://github.com/SujalN/RAGSystemDesign.git
cd RAGSystemDesign
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment Variables

```bash
cp .env.example .env
# fill in:
# OPENAI_API_KEY=
# PINECONE_API_KEY=
# PINECONE_ENV=
# PINECONE_INDEX=
# API_URL=http://localhost:8000
```

---

### 4. Build the Index

Place PDFs in `data/raw/` and run:

```bash
python scripts/build_index.py
```

*Splits files → embeds → upserts to Pinecone in 50‑vector batches.*

---

### 5. Run the API

```bash
uvicorn api.server:app --reload --port 8000
```

*Endpoint:* `POST /chat` – body `{"question": "...", "history": [...]}`

---

### 6. Run the UI

```bash
streamlit run ui/app.py
```

Navigate to [http://localhost:8501](http://localhost:8501).

---

## Usage Examples

```text
You: How many earnings‑call documents do you have indexed?
Bot: I have 10 earnings‑call documents indexed.

You: What risks has Salesforce highlighted over time?
Bot: Salesforce repeatedly cited macro‑economic uncertainty [1] and lengthening sales cycles [2]…

You: Thanks!
Bot: You're welcome! Anything else I can help you with?
```

Click an expander under **Sources** to view the cited snippet.
Press **🗑️ Clear chat** in the **Menu** to start fresh.
