import requests, json, os, textwrap, sys
from typing import List
from dotenv import load_dotenv
load_dotenv()

GROQ_URL  = "https://api.groq.com/openai/v1/chat/completions"
MODEL     = "meta-llama/llama-4-scout-17b-16e-instruct"
HEADERS   = {
    "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
    "Content-Type": "application/json",
}

def _messages(chunks: List[str], question: str) -> List[dict]:
    ctx = "\n\n---\n\n".join(chunks)
    sys = textwrap.dedent(f"""
        You are a helpful assistant. When answering, draw on the contents of <context> and, if needed, supplement with information you find online—label anything sourced from the internet as “from internet” and anything from the document as “from document.” If the required information isn’t available, reply “I don’t know.”
        {ctx}
        </context>
    """).strip()
    return [{"role":"system","content":sys},
            {"role":"user","content":question}]

def chat_stream(chunks: List[str], question: str):
    """Yield the answer tokens as they arrive."""
    payload = {
        "model": MODEL,
        "messages": _messages(chunks, question),
        "temperature": 0.2,
        "stream": True                    # ← magic flag
    }
    with requests.post(GROQ_URL, headers=HEADERS,
                       data=json.dumps(payload), stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or line == b'data: [DONE]':
                continue
            delta = json.loads(line.lstrip(b'data: '))
            token = delta["choices"][0]["delta"].get("content", "")
            if token:
                yield token
