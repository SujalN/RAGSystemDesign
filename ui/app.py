# ui/app.py (snippet)
import streamlit as st
import requests
import os
import urllib.parse

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Salesforce Earnings RAG")
mode = st.selectbox("Mode", ["Q&A", "Summarize"])
query = st.text_input("Enter your text here")

if st.button("Run"):
    if mode == "Q&A":
        resp = requests.post(f"{API_URL}/qa", json={"q": query}).json()
        st.subheader("Answer")
        st.write(resp["answer"])

        st.subheader("Sources")
        # if the API returned inline citations
        cites = resp.get("citations")
        if cites:
            for c in cites:
                idx, chunk_id, snippet = c["index"], c["chunk_id"], c["snippet"]
                pdf_name, page = chunk_id.split("_chunk")[0], int(chunk_id.split("_chunk")[1])
                pdf_url = f"/static/{pdf_name}.pdf#page={page+1}"
                with st.expander(f"[{idx}] {pdf_name}, p.{page+1}"):
                    st.write(snippet)
                    st.markdown(f"[View more in PDF]({pdf_url})")
        else:
            # fallback to old `sources` list
            for src in resp.get("sources", []):
                st.write(f"- {src}")

    else:
        resp = requests.post(f"{API_URL}/summarize", json={"q": query}).json()
        st.subheader("Summary")
        st.write(resp["summary"])
