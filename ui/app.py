import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Salesforce Earnings RAG",
    page_icon=":cloud:",
    layout="wide"
)

def run_query_api(endpoint: str, text: str) -> dict:
    # Send a POST to the given endpoint with the user's query
    try:
        r = requests.post(f"{API_URL}/{endpoint}", json={"q": text})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

tabs = st.tabs(["💬 Q&A", "📋 Summarize"])

# Q&A tab
with tabs[0]:
    st.subheader("Ask a question")
    
    # Text input for Q&A queries
    query_qa = st.text_input(
        "Enter your question…",
        key="query_qa",
        placeholder="What risks did Salesforce highlight in the earnings calls?"
    )
    
    # Button to trigger the Q&A call
    if st.button("Run Q&A", key="run_qa"):
        resp = run_query_api("qa", query_qa)
        if "error" in resp:
            st.error(resp["error"])
        else:
            st.markdown("**Answer:**")
            st.write(resp["answer"])
            st.markdown("**Sources:**")
            for c in resp.get("citations", []):
                idx, chunk_id, snippet = c["index"], c["chunk_id"], c["snippet"]
                
                # Parse file name and page number from the chunk_id
                pdf, pg = chunk_id.split("_chunk")[0], int(chunk_id.split("_chunk")[1])
                
                # Use an expander so the snippet is collapsible
                with st.expander(f"[{idx}] {pdf}, p.{pg+1}"):
                    st.write(snippet)

# Summarize tab
with tabs[1]:
    st.subheader("Generate a summary")
    
    # Text input for summarization queries
    query_sum = st.text_input(
        "Enter topic or keyword…",
        key="query_sum",
        placeholder="e.g. risks over time"
    )
    
    # Button to trigger the summarization call
    if st.button("Run Summarize", key="run_sum"):
        resp = run_query_api("summarize", query_sum)
        if "error" in resp:
            st.error(resp["error"])
        else:
            st.markdown("**Summary:**")
            st.write(resp["summary"])
            st.markdown("**Sources:**")
            for c in resp.get("citations", []):
                idx, chunk_id, snippet = c["index"], c["chunk_id"], c["snippet"]
                
                # Same parsing logic for chunk_id as above
                pdf, pg = chunk_id.split("_chunk")[0], int(chunk_id.split("_chunk")[1])
                url = f"/static/{pdf}.pdf#page={pg+1}"
                with st.expander(f"[{idx}] {pdf}, p.{pg+1}"):
                    st.write(snippet)

