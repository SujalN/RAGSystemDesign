"""
FastAPI back‑end that exposes a conversational RAG endpoint
against Salesforce earnings‑call PDFs stored in Pinecone.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Environment variables
load_dotenv()
PC_API_KEY = os.getenv("PINECONE_API_KEY")
PC_ENV     = os.getenv("PINECONE_ENV")
PC_INDEX   = os.getenv("PINECONE_INDEX")

# Vector store
pc = Pinecone(api_key=PC_API_KEY)
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(
    index_name=PC_INDEX,
    embedding=embeddings,
    pinecone_api_key=PC_API_KEY
)

# Prompt templates
SYSTEM_PROMPT = (
    "You are an expert analyst of Salesforce earnings calls. "
    "Answer ONLY from the provided snippets and cite each fact like [1]. "
    "For greetings or thanks, reply politely without citations."
)
USER_PROMPT = (
    "Snippets:\n{context}\n\n"
    "Question: {question}\n"
    "Answer concisely with inline citations."
)
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(USER_PROMPT),
    ]
)

# Retrieval‑augmented conversational chain
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1) # Low temperature for more deterministic results
conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 8}), # Return top 8 most relevant snippets
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True
)

# FastAPI app and schema
app = FastAPI(title="Salesforce Earnings RAG Chat")

class ChatRequest(BaseModel):
    question: str
    history: List[Tuple[str, str]] = []

class ChatResponse(BaseModel):
    answer: str
    history: List[Tuple[str, str]]
    citations: List[dict]

# Helpers: casual detector
CASUAL_RE = re.compile(r"^(thanks?|thank you|cool|great|awesome|wow|ok|okay)\W*$", re.I)
def is_casual(msg) -> bool:
    return bool(CASUAL_RE.match(msg.strip()))

def count_pdfs() -> int:
    return len(list(Path("data/raw").glob("*.pdf")))

def pages_in_latest() -> int:
    pdfs = sorted(Path("data/raw").glob("*.pdf"))
    if not pdfs:
        return 0
    import PyPDF2
    return len(PyPDF2.PdfReader(str(pdfs[-1])).pages) # Find the page count of most recent upload

# Helper: meta answer
META_QUERIES = {
    r"how many (earnings )?call documents": lambda: (f"I have **{count_pdfs()}** earnings‑call documents indexed.", []),
    r"how many pages.*most recent":          lambda: (f"The most recent call PDF has **{pages_in_latest()}** pages.", []),
}
def maybe_meta_answer(q: str):
    for pat, func in META_QUERIES.items():
        if re.search(pat, q.lower()):
            return func()
    return None

# Endpoint for /chat
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        # 1. Greetings / thanks → polite reply
        if is_casual(req.question):
            polite = "You're welcome! Anything else I can help you with?"
            new_hist = req.history + [(req.question, polite)]
            return ChatResponse(answer=polite, history=new_hist, citations=[])

        # 2. Meta queries → programmatic answer
        meta = maybe_meta_answer(req.question)
        if meta:
            answer, cites = meta
            new_hist = req.history + [(req.question, answer)]
            return ChatResponse(answer=answer, history=new_hist, citations=cites)

        # 3. Otherwise → RAG answer with citations
        result = conv_chain({"question": req.question, "chat_history": req.history})
        answer = result["answer"]
        docs   = result["source_documents"]
        citations = [
            {
                "source": d.metadata.get("source"), 
                "page": d.metadata.get("page"), 
                "content": d.page_content[:500]
             }
            for d in docs
        ]
        new_hist = req.history + [(req.question, answer)]
        return ChatResponse(answer=answer, history=new_hist, citations=citations)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
