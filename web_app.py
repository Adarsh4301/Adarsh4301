import streamlit as st
from uploader   import ingest_pdf
from retriever  import fetch_relevant_chunks
from llm_client import chat_stream
from vector_store import client, COLLECTION

st.set_page_config(page_title="PDF-RAG Demo", page_icon="📚", layout="centered")

# ---------- simple CSS ----------
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

# ---------- helper to list filenames ----------
@st.cache_data(show_spinner=False)
def list_files() -> list[str]:
    pts, _ = client.scroll(COLLECTION, with_payload=True, limit=1000)
    names  = {p.payload.get("file", "Unknown") for p in pts}
    return sorted(names)

# ---------- 1) Upload / index ----------
with st.expander("➕ Upload PDF", expanded=False):
    up = st.file_uploader("Choose a PDF", type="pdf")
    if up and st.button("Index now 🚀"):
        with st.spinner("Indexing …"):
            ingest_pdf(up)
            list_files.clear()      # refresh cache
        st.success("Done! Ask away ↓")

st.divider()

# ---------- 2) Choose file ----------
file_choice = st.selectbox("Search which PDF?", ["All"] + list_files())

# ---------- 3) Question form ----------
with st.form("ask"):
    q = st.text_input("Ask a question")
    run = st.form_submit_button("🔍 Search")

if run and q:
    file_filter = None if file_choice == "All" else file_choice
    hits = fetch_relevant_chunks(q, k=5, file=file_filter)
    ctx  = [h.payload["text"] for h in hits]

    box = st.empty()
    box.markdown('<div class="answer-box">⌛ thinking <span class="spinner">|</span></div>',
                 unsafe_allow_html=True)

    buf = ""
    for tok in chat_stream(ctx, q):
        buf += tok
        box.markdown(f'<div class="answer-box">{buf}▌</div>', unsafe_allow_html=True)
    box.markdown(f'<div class="answer-box">{buf}</div>', unsafe_allow_html=True)

    st.subheader("Sources")
    for h in hits:
        preview = h.payload["text"][:90].replace("\n"," ") + "…"
        fname   = h.payload.get("file", "Unknown")
        page    = h.payload.get("page", "?")
        st.markdown(
            f"- **{fname} – p.{page}** &nbsp;(score {h.score:.2f})<br>{preview}",
            unsafe_allow_html=True
        )
