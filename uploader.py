"""
uploader.py
Save an uploaded file-like object to /tmp and index it.
"""

import uuid, pathlib, tempfile, os
from streamlit.runtime.uploaded_file_manager import UploadedFile
from index_builder import main as build_index


def ingest_pdf(upload: UploadedFile) -> str:
    """Return the collection name we indexed into (always 'docs')."""
    # 1) Write to a temp file
    tmp_dir  = pathlib.Path(tempfile.gettempdir())
    tmp_name = f"{uuid.uuid4()}.pdf"
    tmp_path = tmp_dir / tmp_name
    with tmp_path.open("wb") as f:
        f.write(upload.getbuffer())

    # 2) Run the existing index builder
    build_index(str(tmp_path))

    # 3) Clean up (optional – Qdrant has the data now)
    os.remove(tmp_path)
    return "docs"
