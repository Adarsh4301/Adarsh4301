import streamlit as st
from uploader   import ingest_pdf
from retriever  import fetch_relevant_chunks
from llm_client import chat_stream

st.set_page_config(page_title="PDF-RAG Demo", page_icon="📚",
                   layout="centered")

# ---------- basic style ----------
st.markdown(
    """
    <style>
      .answer-box {border:2px solid #4f8bf9;border-radius:8px;
                   padding:1rem;background:#f7fbff}
      .spinner {animation:blink 1s steps(2,start) infinite}
      @keyframes blink{50%{opacity:0}}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 PDF-RAG Demo (Groq)")

# ---------- 1) Upload & Index ----------
with st.expander("➕ Upload PDF", expanded=False):
    up = st.file_uploader("Choose a PDF file", type="pdf")
    if up and st.button("Index now 🚀"):
        with st.spinner("Embedding & inserting …"):
            ingest_pdf(up)
        st.success("Indexed! Ask away ↓")

st.divider()

# ---------- 2) Ask a question ----------
with st.form("ask"):
    q = st.text_input("Ask a question about the uploaded PDF")
    run = st.form_submit_button("🔍 Search")

if run and q:
    hits = fetch_relevant_chunks(q, k=5)
    ctx  = [h.payload["text"] for h in hits]

    box = st.empty()
    buf = ""
    box.markdown('<div class="answer-box">⌛ thinking <span class="spinner">|</span></div>',
                 unsafe_allow_html=True)

    for tok in chat_stream(ctx, q):
        buf += tok
        box.markdown(f'<div class="answer-box">{buf}▌</div>', unsafe_allow_html=True)

    box.markdown(f'<div class="answer-box">{buf}</div>', unsafe_allow_html=True)

    st.subheader("Sources")

    for h in hits:
     preview = h.payload["text"][:100].replace("\n", " ") + "…"
     page = h.payload.get("page", "?")            # tolerate missing key
     st.markdown(
        f"- p.{page} • <b>{h.score:.2f}</b> – {preview}",
        unsafe_allow_html=True
     )

