"""
Offline script to:
1. Load Salesforce earnings PDFs
2. Split into overlapping text chunks
3. Embed with OpenAI
4. Upsert to Pinecone in safe 50‑vector batches
"""

import logging
import os
from dotenv import load_dotenv
from tqdm import tqdm
from pinecone import Pinecone
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Silence PDF warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Environment variables
load_dotenv()
API_KEY = os.getenv("PINECONE_API_KEY")
INDEX   = os.getenv("PINECONE_INDEX")
DATA    = "data/raw"

# Pinecone client and index
pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX)

# Load PDFs
loader = DirectoryLoader(DATA, glob="*.pdf", loader_cls=PyPDFLoader)
docs   = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks   = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")

# Embed and upsert in batches
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(index_name=INDEX, embedding=embeddings, pinecone_api_key=API_KEY)

BATCH = 50
for i in tqdm(range(0, len(chunks), BATCH), desc="Upserting"):
    vectorstore.add_documents(chunks[i : i + BATCH], batch_size=BATCH)

print(f"Indexed {len(chunks)} chunks into Pinecone index “{INDEX}”")