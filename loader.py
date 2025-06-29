
from pathlib import Path
from typing import List
import PyPDF2  


def load_pdf(path: str | Path) -> List[str]:
    """Return a list of page texts stripped of blank lines."""
    reader = PyPDF2.PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        # Collapse multiple newlines; strip whitespace
        cleaned = "\n".join(
            line.strip() for line in text.splitlines() if line.strip()
        )
        pages.append(cleaned)
    return pages


if __name__ == "__main__":        # quick manual test
    from pprint import pprint
    sample = "abc.pdf"         # drop any PDF next to this file
    out = load_pdf(sample)[:2]    # first 2 pages
    pprint(out)
