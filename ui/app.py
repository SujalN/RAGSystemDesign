"""
Streamlit front‑end for Salesforce Earnings RAG Chat.
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv

# ── Config --------------------------------------------------------------------
load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Salesforce Earnings Chat", page_icon=":cloud:", layout="wide")

# Title + clear‑chat button aligned --------------------------------------------
title_col, btn_col = st.columns([0.85, 0.15])
with title_col:
    st.title("💬 Salesforce Earnings RAG Chat")
with btn_col:
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.history = []
        (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()

# ── Session state -------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []   # [(user, assistant)]

# ── Helper to call API --------------------------------------------------------
def chat_backend(question: str, history):
    resp = requests.post(f"{API_URL}/chat", json={"question": question, "history": history}, timeout=120)
    resp.raise_for_status()
    return resp.json()

# ── Render existing conversation (hide old sources) ---------------------------
if st.session_state.history:
    *earlier, (last_user, last_bot) = st.session_state.history
    for u, a in earlier:
        st.chat_message("user").write(u)
        st.chat_message("assistant").write(a)
    st.chat_message("user").write(last_user)
    prev_assistant = st.chat_message("assistant")
    prev_assistant.write(last_bot)
else:
    prev_assistant = None

# ── Chat input ----------------------------------------------------------------
user_input = st.chat_input("Ask about Salesforce earnings…")

if user_input:
    st.chat_message("user").write(user_input)

    try:
        resp = chat_backend(user_input, st.session_state.history)
    except Exception as e:
        st.error(f"API error: {e}")
    else:
        # Update history
        st.session_state.history = resp["history"]

        # Freeze previous assistant bubble (text only)
        if prev_assistant:
            prev_assistant.empty()
            prev_assistant.write(last_bot)

        # New assistant answer + top‑4 sources
        assistant = st.chat_message("assistant")
        assistant.write(resp["answer"])
        if resp["citations"]:
            assistant.markdown("**Sources:**")
            for i, c in enumerate(resp["citations"][:4], start=1):
                with st.expander(f"[{i}] {c['source']} (p.{c.get('page')})"):
                    st.write(c["content"])
