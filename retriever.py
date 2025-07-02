"""
Retrieve top-k chunks from Qdrant.
Optional arg `file_filter` lets you restrict search to a single PDF.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import (
    ScoredPoint,
    Filter,
    FieldCondition,
    MatchValue,
)

from embeddings import embed

# ───────────────────────── Qdrant connection ────────────────────────────────
client = QdrantClient(
    url     = os.getenv("QDRANT_URL"),      # e.g. https://xyz.cloud.qdrant.io
    api_key = os.getenv("QDRANT_API_KEY"),
)

COLLECTION = "docs"
TOP_K      = 5


# ────────────────────────── main fetch helper ───────────────────────────────
def fetch_relevant_chunks(
    question: str,
    k: int = TOP_K,
    file_filter: Optional[str] = None,
) -> List[ScoredPoint]:
    """
    1. Embed the question.
    2. Search the `docs` collection, optionally restricted to one PDF.
    3. Return raw ScoredPoint objects (score + payload).
    """
    q_vec = embed(question)

    # Optional filename filter
    filt = None
    if file_filter and file_filter != "All":
        filt = Filter(must=[
            FieldCondition(key="file", match=MatchValue(value=file_filter))
        ])

    # NOTE: use query_filter (not filter) with `.search()`
    hits = client.search(
        collection_name = COLLECTION,
        query_vector    = q_vec,
        limit           = k,
        with_payload    = True,
        query_filter    = filt,        # ← fixed argument name
    )
    return hits


# ───────────────────── helper to populate the dropdown ──────────────────────
def list_files() -> List[str]:
    """Return a sorted list of distinct filenames stored in Qdrant."""
    files, offset = set(), None
    while True:
        batch, offset = client.scroll(
            collection_name = COLLECTION,
            limit           = 256,
            with_payload    = {"include": ["file"]},
            offset          = offset,
        )
        for pt in batch:
            if "file" in pt.payload:
                files.add(pt.payload["file"])
        if offset is None:
            break
    return sorted(files)


# ───────────────────────── CLI quick-test block ─────────────────────────────
if __name__ == "__main__":
    import sys
    from tabulate import tabulate

    if len(sys.argv) < 2:
        print("Usage: python app/retriever.py \"Your question\" [file.pdf]")
        sys.exit(1)

    q   = sys.argv[1]
    pdf = sys.argv[2] if len(sys.argv) > 2 else None

    res = fetch_relevant_chunks(q, file_filter=pdf)
    rows = [(h.payload.get("file","?"), h.payload.get("page","?"),
             round(h.score,3), h.payload["text"][:70]+"…")
            for h in res]
    print(tabulate(rows, headers=["file","pg","score","preview"]))
