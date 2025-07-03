# app/vector_store.py
import os, uuid
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

COLLECTION = "docs"
DIM        = 384

# ---------------------- client ----------------------
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# ---------------------- collection ------------------
if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    )

# ---------------------- ensure "file" index ---------
def _ensure_file_index() -> None:
    try:
        # Newer SDK (≥ 1.7) --------------------------
        from qdrant_client.models import PayloadIndexParams, PayloadSchemaType

        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="file",
            field_schema=PayloadIndexParams(type=PayloadSchemaType.KEYWORD),
        )
    except (ImportError, TypeError):
        # Older SDK (≤ 1.6) --------------------------
        try:
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name="file",
                field_type="keyword",       # legacy arg
            )
        except Exception:
            pass  # index already exists or other benign error
    except Exception:
        pass      # index already exists

_ensure_file_index()

# ---------------------- upsert helper ---------------
def upsert_text(text: str, vector: list[float], payload: dict) -> None:
    """
    Insert one chunk into Qdrant.
    `payload` must include at least {"text": ..., "file": ...}
    """
    point_id = uuid.uuid4().int >> 96
    client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(id=point_id, vector=vector, payload=payload)],
    )
