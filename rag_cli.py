
import sys
from retriever import fetch_relevant_chunks
from llm_client import chat

def main() -> None:
    question = input("Question: ").strip()
    if not question:
        print("No question given.")
        return

    hits   = fetch_relevant_chunks(question, k=5)
    chunks = [h.payload["text"] for h in hits]

    answer = chat(chunks, question)

    print("\nAnswer:\n")
    print(answer)
    print("\nSources:")
    for h in hits:
        preview = h.payload["text"][:60].replace("\n", " ") + "…"
        print(f"- score {h.score:.3f}  {preview}")

if __name__ == "__main__":
    main()
