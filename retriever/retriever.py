# retriever/retriever.py
import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# OpenAI v1 client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pinecone client & index
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

class Retriever:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def embed_query(self, query: str):
        resp = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        return resp.data[0].embedding

    def retrieve(self, query: str):
        # 1. Embed the query
        q_vec = self.embed_query(query)

        # 2. Query Pinecone using `vector=` (not `vectors=`)
        results = index.query(
            vector=q_vec,
            top_k=self.top_k,
            include_metadata=True
        )

        # 3. Extract (id, score, snippet)
        return [
            (match["id"], match["score"], match["metadata"].get("snippet", ""))
            for match in results["matches"]
        ]

if __name__ == "__main__":
    r = Retriever()
    docs = r.retrieve("When was the most recent earnings call?")
    for doc_id, score, snippet in docs:
        print(f"{doc_id} (score {score:.4f}): {snippet[:100]}…")
