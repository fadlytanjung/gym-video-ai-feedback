Gym Video AI Feedback API
==========================

Overview
--------
This FastAPI service analyzes uploaded squat videos to extract form metrics (knee & trunk angles), retrieves relevant coaching tips from a static corpus via FAISS, and streams back corrective feedback using ChatGPT. It also provides a general `/ai/chat` endpoint with optional RAG context injection.

Folder Structure
----------------
app/
  core/            ← configuration and OpenAI client  
  db/data.py       ← static tips corpus  
  helpers/
    langchain.py   ← unified RAG + chat streaming helper  
  routers/
    ai.py          ← `/ai/chat` endpoint  
    video.py       ← `/video/feedback` endpoint  
  services/
    context.py     ← FAISS setup & retrieval functions  
    memory.py      ← (optional) LangChain memory helpers  
    video.py       ← MediaPipe pose analysis  
  schemas.py       ← Pydantic request/response models  
  main.py          ← FastAPI app and router registration  

public/
  sample-video.mp4     ← example squat video  
  not-gym-video.mp4    ← unrelated test video  

.env                 ← environment variables  
requirements.txt     ← pip dependencies  
README.txt           ← this file  

Prerequisites
-------------
• Python 3.10+  
• (Optional) Redis for server-side session memory  

Setup & Installation
--------------------
1. Clone repository  
git clone <repo-url> gym-video-ai-feedback
cd gym-video-ai-feedback

cpp
Always show details

Copy

2. Create and activate a virtual environment  
python -m venv .venv
source .venv/bin/activate

markdown
Always show details

Copy

3. Install dependencies  
pip install -r requirements.txt

markdown
Always show details

Copy

4. Copy `.env.example` to `.env` and set your OpenAI API key:  
OPENAI_API_KEY=sk-...
(Optional) REDIS_URL=redis://localhost:6379/0
sql
Always show details

Copy

Running the Server
------------------
Start the FastAPI app with Uvicorn:
uvicorn app.main:app --reload

markdown
Always show details

Copy
By default, the app listens on http://127.0.0.1:8000

Endpoints
---------

1) **POST /video/feedback**  
   - **Upload**: multipart/form-data, field name `file` (MP4 ≤ 50 MB)  
   - **Response**: streamed SSE (`text/event-stream`) with coaching feedback  
   - **Example**:
     ```
     curl -N -X POST http://127.0.0.1:8000/video/feedback \
       -F "file=@public/sample-video.mp4;type=video/mp4"
     ```

2) **POST /ai/chat**  
   - **Body** (JSON):
     {
       "messages": [
         {"role": "user", "content": "How can I improve my squat?"}
       ]
     }
   - **Response**: streamed SSE with ChatGPT reply (automatic RAG if context‐relevant)  
   - **Example**:
     ```
     curl -N -X POST http://127.0.0.1:8000/ai/chat \
       -H "Content-Type: application/json" \
       -d '{"messages":[{"role":"user","content":"How can I improve my squat?"}]}'
     ```

Session Memory (Optional)
-------------------------
By default, `/ai/chat` is stateless: you must resend full history each call. To enable server‐side sessions:

- Configure Redis and `app/services/memory.py`  
- The router will set a `session_id` cookie and store message history in Redis.  
- Clients then only send new turns; the server maintains earlier context.

Extending the Corpus
--------------------
To add or change your static tip corpus:

1. Edit `app/db/data.py` → `STATIC_TIPS` list.  
2. The FAISS index on startup will automatically include your new tips.

License & Credits
-----------------
MIT License — see LICENSE.md.