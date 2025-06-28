import os, dotenv
from huggingface_hub import InferenceClient

dotenv.load_dotenv()

client = InferenceClient(
    model="sentence-transformers/all-MiniLM-L6-v2",
    
    token=os.getenv("HF_TOKEN"),        # token must allow “Make calls to Inference Providers”
    provider="hf-inference",
)

from diskcache import Cache
cache = Cache(".emb_cache")         # folder on disk

def embed(text: str) -> list[float]:
    if text in cache:
        return cache[text]
    vec = client.feature_extraction(text)
    cache[text] = vec
    return vec

