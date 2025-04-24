import os
import re
from typing import List, Optional, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from openai import OpenAI
from pinecone import Pinecone
from retriever.retriever import Retriever

load_dotenv()

# Clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc            = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
retriever     = Retriever(top_k=5)

app = FastAPI()

# Request models with filters
class QARequest(BaseModel):
    q: str
    quarters: Optional[List[str]] = None
    speakers: Optional[List[str]] = None

class SummarizeRequest(BaseModel):
    q: str
    quarters: Optional[List[str]] = None
    speakers: Optional[List[str]] = None

@app.post("/qa")
async def qa_endpoint(body: QARequest):
    try:
        # Build Pinecone metadata filter from the request
        filt: Dict = {}
        if body.quarters:
            filt["quarter"] = {"$in": body.quarters}
        if body.speakers:
            filt["speaker"] = {"$in": body.speakers}

        # Retrieve top-K chunks with optional filter
        docs = retriever.retrieve(body.q, metadata_filter=filt or None)

        # Build the prompt with snippet indices
        contexts = "\n\n".join(f"[{i}] {text}"
                               for i, (_id, _sc, text) in enumerate(docs))
        prompt = (
            "You are an expert on Salesforce earnings calls.\n"
            "Use the following snippets and answer the question.\n"
            "Annotate each fact with inline citations like [1], [2], etc., "
            "where the number refers to the snippet index.\n\n"
            f"{contexts}\n\n"
            f"Question: {body.q}\n"
            "Answer concisely, with inline citations."
        )

        # Call the LLM
        resp = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role":"system", "content":"You are a helpful assistant."},
                {"role":"user",   "content":prompt}
            ]
        )
        answer = resp.choices[0].message.content

        # Extract citation indices like [0], [1], ...
        cited_idxs = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
        citations = [
            {"index": i, "chunk_id": docs[i][0], "snippet": docs[i][2]}
            for i in cited_idxs if i < len(docs)
        ]

        return {"answer": answer, "citations": citations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize")
async def summarize_endpoint(body: SummarizeRequest):
    try:
        filt: Dict = {}
        if body.quarters:
            filt["quarter"] = {"$in": body.quarters}
        if body.speakers:
            filt["speaker"] = {"$in": body.speakers}

        docs = retriever.retrieve(body.q, metadata_filter=filt or None)
        contexts = "\n\n".join(text for (_id, _sc, text) in docs)
        prompt = (
            f"Summarize the following key points about '{body.q}' "
            "from Salesforce earnings call snippets:\n\n"
            f"{contexts}"
        )

        resp = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role":"system", "content":"You are a helpful assistant."},
                {"role":"user",   "content":prompt}
            ]
        )
        return {"summary": resp.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
