# api/server.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from retriever.retriever import Retriever
import re

# Load .env
load_dotenv()

# Instantiate the v1 OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPI + your Retriever
app = FastAPI()
retriever = Retriever(top_k=5)

class Query(BaseModel):
    q: str

@app.post("/qa")
async def qa_endpoint(body: Query):
    try:
        docs = retriever.retrieve(body.q)
        # build contexts with indices
        contexts = "\n\n".join(f"[{i}] {text}"
                               for i, (_id, _sc, text) in enumerate(docs))
        prompt = (
            "You are an expert on Salesforce earnings calls.\n"
            "Use the snippets below and answer the question.\n"
            "Annotate each fact with inline citations like [1], [2], etc.,\n"
            "where the number refers to the snippet index.\n\n"
            f"{contexts}\n\n"
            f"Question: {body.q}\n"
            "Answer concisely, with inline citations."
        )

        resp = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt}
            ]
        )
        answer = resp.choices[0].message.content

        # extract all citation indices like [0], [1], ...
        found = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
        found = sorted(set(found))
        # map them back to docs list
        citations = [
            {"index": i,
             "chunk_id": docs[i][0],
             "snippet": docs[i][2]}
            for i in found
            if i < len(docs)
        ]

        return {"answer": answer, "citations": citations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize")
async def summarize_endpoint(body: Query):
    try:
        docs = retriever.retrieve(body.q)
        contexts = "\n\n".join(text for _id, _sc, text in docs)
        prompt = (
            f"Summarize the following key points about '{body.q}'"
            " from Salesforce earnings call snippets:\n\n"
            f"{contexts}"
        )

        resp = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt}
            ]
        )

        return {"summary": resp.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

