import os
from dotenv import load_dotenv
from typing import Optional, Dict, List

from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# Clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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

    def retrieve(
        self,
        query: str,
        metadata_filter: Optional[Dict] = None
    ) -> List[tuple]:
        q_vec = self.embed_query(query)
        query_args = {
            "vector": q_vec,
            "top_k": self.top_k,
            "include_metadata": True
        }
        if metadata_filter:
            query_args["filter"] = metadata_filter

        resp = index.query(**query_args)
        return [
            (m["id"], m["score"], m["metadata"].get("snippet",""))
            for m in resp["matches"]
        ]

if __name__ == "__main__":
    r = Retriever()
    docs = r.retrieve(
        "most recent earnings call",
        {"quarter": {"$in": ["Q2-2025"]}}
    )
    for doc_id, score, snippet in docs:
        print(f"{doc_id} ({score:.4f}): {snippet[:100]}…")
