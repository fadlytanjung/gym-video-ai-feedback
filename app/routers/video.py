import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from app.services.video import analyze_video
from app.services.context import retrieve_tips
from app.helpers.langchain import generate_stream

router = APIRouter(prefix="/video", tags=["video"])
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

@router.post("/feedback")
async def video_feedback(file: UploadFile = File(...)):
    tmp_path = f"/tmp/{uuid.uuid4()}.mp4"
    bytes_written = 0

    try:
        with open(tmp_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > MAX_FILE_SIZE:
                    raise HTTPException(413, "File too large")
                out.write(chunk)

        metrics = analyze_video(tmp_path)
        if metrics is None:
            raise HTTPException(400, "No squat detected")

        tips = retrieve_tips(metrics, k=3)

        metric_msg = {
            "role": "user",
            "content": (
                f"Athlete metrics:\n"
                f"- Knee angle: {metrics['knee_angle']:.1f}°\n"
                f"- Trunk lean: {metrics['trunk_angle']:.1f}°"
            )
        }

        return StreamingResponse(
            generate_stream(
                user_history=[metric_msg],
                tips=tips,
                rag_system_prompt="You are an expert squat coach.",
                rag_prefix="Here are some coaching tips for this squat:\n",
                rag_suffix="\nPlease give friendly, corrective feedback:",
            ),
            media_type="text/plain; charset=utf-8",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
