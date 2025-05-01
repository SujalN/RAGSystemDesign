import os
import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from retriever.retriever import Retriever

load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Retriever (wraps Pinecone + embeddings)
retriever = Retriever(top_k=5)

app = FastAPI()

class QARequest(BaseModel):
    # Incoming payload for Q&A requests.
    q: str

class SummarizeRequest(BaseModel):
    # Incoming payload for summarization requests.
    q: str
    
def extract_citation_indices(text: str) -> list[int]:
    # From output containing inline citations, return a sorted list of unique integer indices
    found = re.findall(r"\[(\d+)\]", text)
    return sorted({int(n) for n in found})

@app.post("/qa")
async def qa_endpoint(body: QARequest):
    try:
        # Fetch relevant chunks (id, score, snippet)
        docs = retriever.retrieve(body.q)
        
        # Build context section of the prompt
        contexts = "\n\n".join(f"[{i}] {text}"
                               for i, (_id,_sc,text) in enumerate(docs))
        
        # Compose prompt
        prompt = (
            "You are an expert on Salesforce earnings calls.\n"
            "Use the following snippets to answer the question.\n"
            "Annotate each fact with inline citations like [1], [2], etc., "
            "where the number refers to the snippet index.\n\n"
            f"{contexts}\n\n"
            f"Question: {body.q}\n"
            "Answer concisely, with inline citations."
        )
        
        # Call LLM
        resp = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role":"system","content":"You are a helpful assistant."},
                {"role":"user",  "content":prompt}
            ]
        )
        answer = resp.choices[0].message.content
        
        # Extract and map citations back to chunks
        indices = extract_citation_indices(answer)
        citations = [
            {
                "index": i, 
                "chunk_id": docs[i][0], 
                "snippet": docs[i][2]
             }
            for i in indices if i < len(docs)
        ]
        return {"answer": answer, "citations": citations}

    except Exception as e:
        # Bubble up any error as a 500 with the message
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize")
async def summarize_endpoint(body: SummarizeRequest):
    try:
        docs = retriever.retrieve(body.q)
        contexts = "\n\n".join(f"[{i}] {text}"
                               for i, (_id,_sc,text) in enumerate(docs))
        prompt = (
            "You are an expert on Salesforce earnings calls.\n"
            "Summarize the key points about the topic below. "
            "Annotate each summarized point with inline citations like [1], [2], etc., "
            "where the number refers to the snippet index.\n\n"
            f"{contexts}\n\n"
            f"Topic: {body.q}\n"
            "Provide a concise summary with inline citations."
        )
        resp = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role":"system","content":"You are a helpful assistant."},
                {"role":"user",  "content":prompt}
            ]
        )
        summary = resp.choices[0].message.content
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", summary)})
        citations = [
            {"index": i, "chunk_id": docs[i][0], "snippet": docs[i][2]}
            for i in cited if i < len(docs)
        ]
        return {"summary": summary, "citations": citations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
