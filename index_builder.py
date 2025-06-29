

import time, sys
from pathlib import Path

from embeddings import embed         
from vector_store import upsert_text  
from loader import load_pdf
from text_chunker import split_text
from tqdm import tqdm                 # progress bar

BATCH = 8                 # HF free tier ⟶ keep calls ≤30/min

def main(pdf_path: str):
    pages = load_pdf(pdf_path)
    print(f"Loaded {len(pages)} pages")

    texts = []
    for page in pages:
        texts.extend(split_text(page))

    print(f"Total chunks: {len(texts)}")

    for i in tqdm(range(0, len(texts), BATCH), desc="Embedding"):
        batch = texts[i : i + BATCH]
        vectors = []
        for t in batch:
            vectors.append(embed(t))
            time.sleep(2)            # be polite: 30 calls/min → 2 s gap
        for t, v in zip(batch, vectors):
            upsert_text(t, v)

    print("✓ PDF indexed into Qdrant")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: p.pdfython index_builder.py a")
        sys.exit(1)
    main(sys.argv[1])
