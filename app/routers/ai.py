import uuid
from typing import AsyncGenerator, List, Dict
from fastapi import APIRouter, HTTPException, Cookie, Response
from fastapi.responses import StreamingResponse

from app.schemas import AIRequest
from app.services.context import retrieve_tips_with_scores
from app.helpers.langchain import generate_stream

router = APIRouter(prefix="/ai", tags=["ai"])
SIMILARITY_THRESHOLD = 0.75

_conversations: Dict[str, List[Dict[str, str]]] = {}

@router.post("/chat")
async def chat_endpoint(
    req: AIRequest,
    response: Response,
    session_id: str | None = Cookie(default=None),
):
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            samesite="lax",
        )

    history = _conversations.setdefault(session_id, [])

    for msg in req.messages:
        if msg["role"] == "user":
            history.append(msg)

    last_user = history[-1]["content"]
    scored    = retrieve_tips_with_scores(last_user, k=3)
    tips      = [tip for tip, score in scored if score >= SIMILARITY_THRESHOLD]

    generator = generate_stream(
        user_history=history,
        tips=tips,
        rag_system_prompt="You are an expert gym coach. Use these tips where helpful.",
        rag_prefix="Here are some relevant tips:\n",
        rag_suffix="",
        chat_system_prompt="You are a helpful assistant."
    )

    return StreamingResponse(
        generator,
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
