# RAGSystemDesign
This is a solution that provides Question &amp; Answer (Q&amp;A) and summarization capabilities based on Salesforce's quarterly earnings presentation transcripts.

## Prerequisites
- Python 3.8+  
- `.env` file with OpenAI & Pinecone keys  

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Architecture
salesforce‐rag/  
├── README.md
├── requirements.txt  
├── .gitignore  
├── data/  
│   ├── raw/  
│   └── chunks/  
├── scripts/  
│   ├── convert_pdfs.py  
│   └── prepare_chunks.py  
├── embeddings/  
│   └── indexer.py  
├── retriever/  
│   └── retriever.py  
├── api/  
│   └── server.py  
└── ui/  
    └── app.py 