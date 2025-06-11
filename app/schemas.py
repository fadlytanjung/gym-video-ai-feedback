from fastapi import UploadFile
from pydantic import BaseModel

class AIRequest(BaseModel):
    messages: list[dict]   # [{"role": "user", "content": "..."}]

class AIStreamChunk(BaseModel):
    delta: str
