# 🏋️ Gym Video AI Feedback API

## 📖 Overview
This FastAPI service lets you upload a squat video 📹 and returns live, coach-style feedback 🏋️‍♂️ on your form (knee & trunk angles). It also supports a chat interface with automatic RAG context from a static coaching tip corpus.

## System Architechture

![system-architecture](https://github.com/user-attachments/assets/3a566332-9bbf-4370-aec6-3cb7a399c5e2)


## 📂 Folder Structure

```
app/
├── core/                   # Configuration & OpenAI/Gemini clients
│   ├── config.py           
│   └── ...                 
├── db/                     
│   └── data.py             # `STATIC_TIPS` list (your RAG corpus)
├── helpers/                
│   └── langchain.py        # Unified RAG + chat streaming helper
├── routers/                
│   ├── ai.py               # `/ai/chat` endpoint
│   └── video.py            # `/video/feedback` endpoint  
├── services/               
│   ├── context.py          # FAISS index & retrieval logic  
│   ├── memory.py           # (optional) session memory helpers  
│   └── video.py            # MediaPipe pose detection & angle computation  
├── schemas.py              # Pydantic models for requests  
└── main.py                 # FastAPI app setup & CORS configuration  
public/                     # Example videos  
├── sample-video.mp4        
└── not-gym-video.mp4
.env                        # Environment variables  
requirements.txt            # Python dependencies  
README.md                   # This file
```

## ⚙️ Setup & Installation

1. **Clone** the repository  
   ```bash
   git clone https://github.com/fadlytanjung/gym-video-ai-feedback.git
   cd gym-video-ai-feedback
   ```

2. **Create & activate** a virtual environment  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install** dependencies  
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure** environment variables  
   ```bash
   cp .env.example .env
   # Edit `.env`:
   OPENAI_API_KEY=sk-...
   # (Optional) REDIS_URL=redis://localhost:6379/0
   ```

## 🚀 Running the Server

```bash
uvicorn app.main:app --reload
```

By default, it listens on **http://127.0.0.1:8000**

## 🔌 Endpoints

### 1. POST `/video/feedback`  
Upload a squat video and stream back live feedback.

- **Form-data**: field `file` (MP4 ≤ 50 MB)  
- **Response**: plain-text stream (`text/plain`)  
- **Example**:
  ```bash
  curl -N -X POST http://127.0.0.1:8000/video/feedback \
    -F "file=@public/sample-video.mp4;type=video/mp4"
  ```

### 2. POST `/ai/chat`  
Send a chat message; receives a streamed plain-text reply with optional RAG tips.

- **JSON**:  
  ```json
  { "messages": [
      { "role": "user", "content": "How can I improve my squat?" }
    ]
  }
  ```
- **Response**: plain-text stream  
- **Example**:
  ```bash
  curl -N -X POST http://127.0.0.1:8000/ai/chat \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"How can I improve my squat?"}]}'
  ```

## 🖥 Frontend Integration

Check out the React/Next.js client at:
https://github.com/fadlytanjung/chatbot-ai

## 🔄 Session Memory (Optional)

By default, `/ai/chat` is stateless. To enable server-side sessions:
- Configure Redis in `app/services/memory.py`  
- The router sets a `session_id` cookie and stores message history in Redis.

## 📝 Extending the Corpus

To customize your coaching tips:
1. Edit `app/db/data.py` → `STATIC_TIPS` list.  
2. Restart the server to rebuild the FAISS index.

## 📜 License

MIT License — see [LICENSE.md](LICENSE.md)
