# app/retriever.py
"""
Retrieve top-k chunks from Qdrant.
If `file` is provided, the search is restricted to that PDF only.
Compatible with both old (query_filter=) and new (filter=) SDK versions.
"""
import os
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from embeddings import embed

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

COLLECTION = "docs"


# ----------------------------------------------------------------------
# Internal helper: choose filter= or query_filter= depending on SDK
# ----------------------------------------------------------------------
def _search(q_vec, filt, k: int) -> List[ScoredPoint]:
    try:
        # ≥ 1.6 — accepts `filter=`
        return client.search(
            collection_name=COLLECTION,
            query_vector=q_vec,
            limit=k,
            with_payload=True,
            filter=filt,
        )
    except AssertionError:
        # < 1.6 — falls back to `query_filter=`
        return client.search(
            collection_name=COLLECTION,
            query_vector=q_vec,
            limit=k,
            with_payload=True,
            query_filter=filt,
        )


def fetch_relevant_chunks(
    question: str,
    k: int = 5,
    file: Optional[str] = None,
) -> List[ScoredPoint]:
    """
    Embed `question`, then return the top-k most similar chunks.
    If `file` is not None (and not "All"), restrict to that PDF.
    """
    q_vec = embed(question)

    filt = None
    if file and file.upper() != "__ALL__":
        filt = {"must": [{"key": "file", "match": {"value": file}}]}

    return _search(q_vec, filt, k)


# ------------------ quick CLI test ------------------
if __name__ == "__main__":
    import sys, tabulate

    q = " ".join(sys.argv[1:]) or "Sample question"
    rows = [
        (h.payload.get("file"), round(h.score, 3), h.payload["text"][:70] + "…")
        for h in fetch_relevant_chunks(q)
    ]
    print(tabulate.tabulate(rows, headers=["file", "score", "preview"]))
