"""
Smoke-test:
1. Embed one sentence with Hugging Face InferenceClient
2. Upsert the vector into your Qdrant Cloud collection
"""

import sys
from embeddings import embed               # local module in the same folder
from vector_store import upsert_text       # helper we just defined

def main() -> None:
    text = "Fast language models unlock real-time products."

    # ①  Embedding
    try:
        vector = embed(text)
    except Exception as exc:
        print("‼️  Embedding call failed:", exc, file=sys.stderr)
        print("Check HF_TOKEN permissions ➜ Settings → Access tokens → ✓ Make calls to Inference Providers")
        return
    print("Vector len =", len(vector))      # should print 384

    # ②  Qdrant insert
    try:
        upsert_text(text, vector)
    except Exception as exc:
        print("‼️  Qdrant upsert failed:", exc, file=sys.stderr)
        print("Verify QDRANT_URL / QDRANT_API_KEY in .env and that the collection exists.")
        return

    print("✓ Saved to Qdrant!")

if __name__ == "__main__":
    main()
