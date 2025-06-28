# app/text_chunker.py
"""
Simple character-based splitter.
500 characters ≈ ~100-120 words ≈ good context chunk
"""

def split_text(text: str, *, size: int = 500, overlap: int = 50):
    start = 0
    while start < len(text):
        end = start + size
        yield text[start:end]
        start = end - overlap
