"""
Embed every chunk of a PDF and batch-upload to Qdrant.
Usage:  python app/index_builder.py docs/abc.pdf
"""
import sys, time, uuid
from pathlib import Path
from tqdm import tqdm

from embeddings   import embed
from vector_store import client, COLLECTION
from loader       import load_pdf
from text_chunker import split_text

BATCH = 32

def main(pdf_path: str):
    pdf_name = Path(pdf_path).name            # e.g. "abc.pdf"
    pages    = load_pdf(pdf_path)
    print(f"Loaded {len(pages)} pages")

    rows = []
    for pg, page in enumerate(pages, start=1):
        for idx, chunk in enumerate(split_text(page)):
            rows.append((chunk, pg, idx, pdf_name))

    print(f"Total chunks: {len(rows)}")

    for i in tqdm(range(0, len(rows), BATCH), desc="Embedding"):
        batch   = rows[i : i + BATCH]
        vectors = [embed(r[0]) for r in batch]
        time.sleep(1)                         # polite pause

        ids     = [uuid.uuid4().int >> 96 for _ in batch]
        payload = [
            {"text": t, "page": pg, "chunk": idx, "file": fname}
            for (t, pg, idx, fname) in batch
        ]

        client.upload_collection(
            collection_name=COLLECTION,
            vectors=vectors,
            payload=payload,
            ids=ids,
            batch_size=BATCH,
        )

    print("✓ PDF indexed into Qdrant")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python app/index_builder.py <PDF_PATH>")
        sys.exit(1)
    main(sys.argv[1])
