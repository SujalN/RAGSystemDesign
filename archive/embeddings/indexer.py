import os
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# OpenAI v1 client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pinecone client & index
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

# Where your chunked .txt files live
CHUNK_DIR = Path(__file__).parent.parent / "data" / "chunks"

def embed_text(text: str):
    resp = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return resp.data[0].embedding

def process_chunk(chunk_file: Path):
    text = chunk_file.read_text(encoding="utf-8")
    vector = embed_text(text)

    # Metadata extraction
    parts = chunk_file.stem.split("_chunk")[0].split("-")
    quarter = parts[-1]
    speaker = "unknown"

    metadata = {
        "source": chunk_file.stem,
        "snippet": text[:200],
        "quarter": quarter,
        "speaker": speaker
    }

    index.upsert([(chunk_file.stem, vector, metadata)])
    print(f"Upserted {chunk_file.name}")

def main():
    chunks = list(CHUNK_DIR.glob("*.txt"))
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(process_chunk, chunks)

if __name__ == "__main__":
    main()
