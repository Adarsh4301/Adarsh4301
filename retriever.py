# app/retriever.py
"""
Given a question string ➜ return top-k chunk texts from Qdrant.
"""
import os
from typing import List
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, ScoredPoint

from embeddings import embed          # same HF embedder you used before

# ---------------- Qdrant client ----------------
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

COLLECTION = "docs"                   # same as in vector_store.py
TOP_K      = 5                        # you can tune later


def fetch_relevant_chunks(question: str, k: int = TOP_K) -> List[ScoredPoint]:
    """
    1. Embed the question text.
    2. Use Qdrant 'search' to get top-k similar points.
    3. Return the raw ScoredPoint objects (include score & payload).
    """
    q_vec = embed(question)           # ① embed question
    hits = client.search(
        collection_name=COLLECTION,
        query_vector=q_vec,
        limit=k,
        with_payload=True             # crucial – we need the chunk text!
    )
    return hits


# ---- quick test when run directly (python app/retriever.py) --------------
if __name__ == "__main__":
    import sys
    from tabulate import tabulate

    if len(sys.argv) < 2:
        print("Usage: python app/retriever.py \"Your question here\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    results = fetch_relevant_chunks(query)

    rows = [
        (round(hit.score, 3), hit.payload["text"][:80] + "…")
        for hit in results
    ]
    print(tabulate(rows, headers=["score", "chunk preview"]))
