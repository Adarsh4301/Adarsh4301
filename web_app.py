import streamlit as st
from retriever import fetch_relevant_chunks
from llm_client import chat_stream

st.title("📚 PDF-RAG Demo (Groq)")

question = st.text_input("Ask a question about your PDF")
if question:
    hits = fetch_relevant_chunks(question)
    ctx  = [h.payload["text"] for h in hits]

    answer = st.empty()
    buf = ""
    for tok in chat_stream(ctx, question):
        buf += tok
        answer.markdown(buf + "▌")

    st.subheader("Sources")
    for h in hits:
        st.markdown(f"- *score {h.score:.3f}*: {h.payload['text'][:80]}…")
