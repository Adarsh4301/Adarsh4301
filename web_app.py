import streamlit as st
from retriever import fetch_relevant_chunks
from llm_client import chat_stream

st.set_page_config(page_title="PDF-RAG Demo", page_icon="📚", layout="centered")

# ---------- simple CSS tweaks ----------
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; }
      .answer-box      { border: 2px solid #4f8bf9; border-radius: 0.5rem;
                         padding: 1rem; background: #f7fbff; }
      .sources li      { line-height: 1.35; margin-bottom: .5rem;}
      .spinner         { font-size:1.2em; animation: blink 1s steps(2, start) infinite; }
      @keyframes blink { 50% { opacity: 0; } }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 PDF-RAG Demo (Groq)")

with st.form("query"):
    question = st.text_input("Ask a question about your PDF")
    submitted = st.form_submit_button("🔍 Search")

if submitted and question:
    # --- retrieval ---
    hits = fetch_relevant_chunks(question, k=5)
    ctx  = [h.payload["text"] for h in hits]

    # --- animated answer box ---
    answer_box = st.empty()
    answer_box.markdown('<div class="answer-box">⌛ thinking&nbsp;<span class="spinner">|</span></div>', unsafe_allow_html=True)

    buf = ""
    for tok in chat_stream(ctx, question):
        buf += tok
        answer_box.markdown(f'<div class="answer-box">{buf}▌</div>', unsafe_allow_html=True)

    # final render (remove cursor)
    answer_box.markdown(f'<div class="answer-box">{buf}</div>', unsafe_allow_html=True)

    # --- cite sources ---
    st.subheader("Sources")
    st.markdown("<ul class='sources'>", unsafe_allow_html=True)
    for h in hits:
        preview = h.payload["text"][:100].replace("\n", " ") + "…"
        st.markdown(f"<li><b>score {h.score:.2f}</b> · {preview}</li>", unsafe_allow_html=True)
    st.markdown("</ul>", unsafe_allow_html=True)
