import os, uuid
from dotenv import load_dotenv; load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),          # https://xyz.cloud.qdrant.io
    api_key=os.getenv("QDRANT_API_KEY"),
)

COLLECTION = "docs"
DIM = 384

if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    )

def upsert_text(text: str, vector: list[float]):
    """Insert one text-vector pair into Qdrant."""
    point_id = uuid.uuid4().int >> 96
    client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(id=point_id,
                            vector=vector,
                            payload={"text": text})],
    )
