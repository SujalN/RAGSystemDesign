# embeddings/indexer.py
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Load env vars
load_dotenv()

# --- OpenAI client setup (v1) ---
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Pinecone client setup ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

# Directory containing your chunk files
CHUNK_DIR = Path(__file__).parent.parent / "data" / "chunks"

def embed_text(text: str):
    # Use the v1 OpenAI client for embeddings
    resp = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    # Extract the embedding vector
    return resp.data[0].embedding

def main():
    for chunk_file in CHUNK_DIR.glob("*.txt"):
        text = chunk_file.read_text(encoding="utf-8")
        vector = embed_text(text)
        metadata = {
            "source": chunk_file.stem,
            "snippet": text[:200]
        }
        print(f"Upserting {chunk_file.name} …")
        index.upsert([(chunk_file.stem, vector, metadata)])

if __name__ == "__main__":
    main()
