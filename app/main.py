from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.video import router as video_router
from app.routers.ai    import router as ai_router

app = FastAPI(title="Gym Feedback API")

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video_router)
app.include_router(ai_router)
