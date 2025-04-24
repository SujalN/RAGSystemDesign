import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Salesforce Earnings RAG", layout="wide")

# Sidebar: filters
st.sidebar.title("Filters")
quarters = st.sidebar.multiselect(
    "Quarter", ["Q1-2024","Q2-2024","Q3-2024","Q4-2024","Q1-2025"]
)
speakers = st.sidebar.multiselect(
    "Speaker", ["CEO","CFO","COO","Analyst"]
)

# Main UI
st.title("Salesforce Earnings RAG")
mode = st.radio("Mode", ["Q&A", "Summarize"], key="mode")

# Session state setup
if "query" not in st.session_state:
    st.session_state.query = ""
if "response" not in st.session_state:
    st.session_state.response = None

# Callback: run when Enter pressed or button clicked
def run_query():
    q = st.session_state.query.strip()
    if not q:
        st.session_state.response = None
        return

    payload = {
        "q": q,
        "quarters": quarters,
        "speakers": speakers
    }
    endpoint = "qa" if st.session_state.mode == "Q&A" else "summarize"
    try:
        st.session_state.response = requests.post(
            f"{API_URL}/{endpoint}",
            json=payload,
        ).json()
    except Exception as e:
        st.session_state.response = {"error": str(e)}

# Text input: on_change fires when user presses Enter
st.text_input(
    label="Enter your question or topic",
    key="query",
    on_change=run_query,
    placeholder="Type here and press Enter…"
)

# Run button
if st.button("Run"):
    run_query()

# Display results
resp = st.session_state.response
if resp:
    if st.session_state.mode == "Q&A":
        st.subheader("Answer")
        st.write(resp.get("answer", resp.get("error", "")))

        st.subheader("Sources")
        for c in resp.get("citations", []):
            idx      = c["index"]
            chunk_id = c["chunk_id"]
            snippet  = c["snippet"]
            pdf_file, page = chunk_id.split("_chunk")[0], int(chunk_id.split("_chunk")[1])
            pdf_url = f"/static/{pdf_file}.pdf#page={page+1}"
            with st.expander(f"[{idx}] {pdf_file}, p.{page+1}"):
                st.write(snippet)
                st.markdown(f"[View more in PDF]({pdf_url})")
    else:  # Summarize
        st.subheader("Summary")
        st.write(resp.get("summary", resp.get("error", "")))
