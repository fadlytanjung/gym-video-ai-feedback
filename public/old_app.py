import mediapipe as mp
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import os, cv2, faiss

# 1) Load your OpenAI key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# 2) Static gym tips (your RAG corpus)
static_tips = [
    "Maintain a neutral spine throughout the squat to protect your lower back.",
    "Drive your knees outward, keeping them aligned with your toes.",
    "Push through your heels to engage your glutes and hamstrings.",
    "Keep your chest up and gaze forward to stabilize your torso.",
    "Brace your core by inhaling before descending to support your spine.",
    "Lower yourself until your thighs are parallel to the floor or below.",
    "Avoid locking your knees at the top to maintain tension.",
    "Control the descent and ascent; avoid bouncing at the bottom.",
    "Distribute weight evenly between both feet for balanced strength.",
    "Ensure full hip and knee extension at the top of each rep."
]

# 3) Embed with OpenAI’s text-embedding-ada-002
def get_embeddings(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    embs = [item.embedding for item in resp.data]
    return np.array(embs, dtype="float32")

# 4) Build FAISS index manually
def build_vector_store(tips: list[str]):
    embeddings = get_embeddings(tips)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# 5) MediaPipe form analysis (unchanged)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    return res.pose_landmarks.landmark if res.pose_landmarks else None

def compute_joint_angle(a, b, c):
    import math
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    return math.degrees(math.acos(dot/mag)) if mag else 0

def analyze_form(landmarks):
    L = mp_pose.PoseLandmark
    hip, knee, ankle = landmarks[L.LEFT_HIP], landmarks[L.LEFT_KNEE], landmarks[L.LEFT_ANKLE]
    shoulder = landmarks[L.LEFT_SHOULDER]
    return {
        'knee_angle': compute_joint_angle(hip, knee, ankle),
        'trunk_angle': compute_joint_angle(shoulder, hip, knee)
    }

# 6) Run video, compute average metrics
def get_average_metrics(video_path: str):
    cap = cv2.VideoCapture(video_path)
    metrics_list = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        idx += 1
        if idx % 30: continue
        lm = extract_landmarks(frame)
        if lm:
            metrics_list.append(analyze_form(lm))
    cap.release()
    if not metrics_list:
        # raise RuntimeError("No pose data found.")
        return None
    avg = lambda k: sum(m[k] for m in metrics_list) / len(metrics_list)
    return {'knee_angle': avg('knee_angle'), 'trunk_angle': avg('trunk_angle')}

# 7) Retrieve top-K tips from FAISS
def retrieve_contexts(index: faiss.Index, tips: list[str], query: str, k: int = 3):
    q_emb = get_embeddings([query])
    D, I = index.search(q_emb, k)
    return [tips[i] for i in I[0]]

# 8) Generate ChatGPT feedback
def generate_feedback_with_chatgpt(metrics: dict, contexts: list[str]) -> str:
    system_prompt = "You are an expert gym coach who provides friendly, detailed squat feedback."
    context_block = "\n".join(f"- {c}" for c in contexts)
    user_prompt = (
        f"Here are some relevant tips:\n{context_block}\n\n"
        f"Athlete metrics:\n"
        f"- Knee angle: {metrics['knee_angle']:.1f}°\n"
        f"- Trunk lean angle: {metrics['trunk_angle']:.1f}°\n\n"
        f"Please give corrective feedback."
    )

    resp = client.chat.completions.create(model="gpt-4o-mini",  # or "gpt-4o", or "gpt-3.5-turbo"
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.7,
    max_tokens=200)
    return resp.choices[0].message.content.strip()

# 9) Main
if __name__ == "__main__":
    # metrics = get_average_metrics("sample-video.mp4")
    metrics = get_average_metrics("not-gym-video.mp4")
    print("---metrics--\n")
    print(metrics)

    if metrics is None:
        # no human squat detected
        print("⚠️  No squat form detected in this video. Please provide a human performing a squat.")
    else:
        # Build and query vector store
        faiss_idx = build_vector_store(static_tips)

        print("---faiss_idx--\n")
        print(faiss_idx)
        # Form a text query based on metrics
        query = f"Tips for knee {metrics['knee_angle']:.1f}° and trunk {metrics['trunk_angle']:.1f}°"
        contexts = retrieve_contexts(faiss_idx, static_tips, query, k=3)

        # Generate and print feedback
        feedback = generate_feedback_with_chatgpt(metrics, contexts)
        print("\n=== AI Coach Feedback (ChatGPT + RAG) ===\n")
        print(feedback)
