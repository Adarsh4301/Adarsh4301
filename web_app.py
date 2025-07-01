import streamlit as st
from uploader import ingest_pdf
from retriever import fetch_relevant_chunks
from llm_client import chat_stream

st.set_page_config(page_title="PDF-RAG Demo",
                   page_icon=":books:", layout="centered")

st.title("📚 PDF-RAG Demo (Groq)")

# -------- 1) UPLOAD FORM --------
with st.expander("➕ Upload & index a PDF", expanded=False):
    upload = st.file_uploader(
        "Choose a PDF",
        type="pdf",
        help="The file is processed only in memory; no data is stored permanently.",
    )
    if upload and st.button("Index now 🚀"):
        with st.spinner("Embedding & inserting into Qdrant …"):
            ingest_pdf(upload)
        st.success("Done! Ask anything about that PDF ↓")

st.divider()

# -------- 2) QUESTION FORM --------
with st.form("ask_form"):
    question = st.text_input("Ask a question")
    submitted = st.form_submit_button("🔍 Search")

if submitted and question:
    # ---- retrieval
    hits = fetch_relevant_chunks(question, k=5)
    ctx  = [h.payload["text"] for h in hits]

    # ---- streaming answer
    placeholder = st.empty()
    buf = ""
    for tok in chat_stream(ctx, question):
        buf += tok
        placeholder.markdown(f"#### Answer\n\n{buf}▌")
    placeholder.markdown(f"#### Answer\n\n{buf}")

    # ---- citations
    st.subheader("Sources")
    for h in hits:
        preview = h.payload["text"][:100].replace("\n", " ") + "…"
        st.markdown(
            f"- p.{h.payload.get('page','?')} ·"
            f" **{h.score:.2f}** &nbsp; {preview}"
        )
