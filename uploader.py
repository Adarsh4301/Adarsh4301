"""uploader.py – save an uploaded PDF to /tmp then feed index_builder."""
import tempfile, pathlib
from streamlit.runtime.uploaded_file_manager import UploadedFile
from index_builder import main as build_index

def ingest_pdf(upload: UploadedFile) -> None:
    tmp_dir  = pathlib.Path(tempfile.gettempdir())
    tmp_path = tmp_dir / upload.name
    with tmp_path.open("wb") as f:
        f.write(upload.getbuffer())

    build_index(str(tmp_path))       # runs the batch indexer
