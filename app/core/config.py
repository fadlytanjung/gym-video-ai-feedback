from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

def get_embeddings(texts: list[str]) -> np.ndarray:
    resp = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    embs = [item.embedding for item in resp.data]
    return np.array(embs, dtype="float32")