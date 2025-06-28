# quick_play.py
from retriever import fetch_relevant_chunks
from llm_client import chat_stream

q = "What is c# ?"
hits = fetch_relevant_chunks(q)
for tok in chat_stream([h.payload["text"] for h in hits], q):
    print(tok, end="", flush=True)
print()
