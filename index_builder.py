
import sys, time, uuid
from pathlib import Path
from tqdm import tqdm

from embeddings   import embed
from vector_store import client, COLLECTION         # uses same client
from loader       import load_pdf
from text_chunker import split_text

BATCH = 32                       # nice for free-tier HF

def main(pdf_path: str):
    pages = load_pdf(pdf_path)
    print(f"Loaded {len(pages)} pages")

    chunks = []
    for page_num, page in enumerate(pages, start=1):
        for chunk_idx, chunk in enumerate(split_text(page)):
            chunks.append((chunk, page_num, chunk_idx))

    print(f"Total chunks: {len(chunks)}")

    for i in tqdm(range(0, len(chunks), BATCH), desc="Embedding"):
        batch = chunks[i : i + BATCH]

        # ---- 1) embed
        vectors = [embed(c[0]) for c in batch]      # index 0 = text
        time.sleep(1)                               # polite

        # ---- 2) build parallel lists for upload_collection
        ids     = [uuid.uuid4().int >> 96 for _ in batch]
        payload = [
            {"text": text, "page": pg, "chunk": idx}
            for (text, pg, idx) in batch
        ]

        client.upload_collection(
            collection_name=COLLECTION,
            vectors=vectors,
            payload=payload,
            ids=ids,
            batch_size=BATCH
        )

    print("✓ PDF indexed into Qdrant")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python app/index_builder.py <PDF_PATH>")
        sys.exit(1)
    main(sys.argv[1])
