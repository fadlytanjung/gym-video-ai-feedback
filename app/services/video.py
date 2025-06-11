from app.services.context import retrieve_tips, build_rag_messages
from typing import AsyncGenerator, Optional
import mediapipe as mp
import numpy as np
import faiss, cv2

mp_pose = mp.solutions.pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5
)

def analyze_video(video_path: str) -> Optional[dict]:
    """
    Run MediaPipe on the saved file, compute avg squat metrics.
    Returns None if no human pose is detected.
    """
    cap = cv2.VideoCapture(video_path)
    metrics_list = []
    frame_i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_i += 1
        if frame_i % 30 != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_pose.process(rgb)
        lm = res.pose_landmarks.landmark if res.pose_landmarks else None
        if lm:
            metrics_list.append(_compute_angles(lm))

    cap.release()

    if not metrics_list:
        return None

    return {
        "knee_angle": np.mean([m["knee_angle"] for m in metrics_list]),
        "trunk_angle": np.mean([m["trunk_angle"] for m in metrics_list]),
    }

def _compute_angles(landmarks) -> dict:
    import math
    L = mp.solutions.pose.PoseLandmark

    hip      = landmarks[L.LEFT_HIP]
    knee     = landmarks[L.LEFT_KNEE]
    ankle    = landmarks[L.LEFT_ANKLE]
    shoulder = landmarks[L.LEFT_SHOULDER]

    def angle(a, b, c):
        ba = (a.x - b.x, a.y - b.y)
        bc = (c.x - b.x, c.y - b.y)
        dot = ba[0]*bc[0] + ba[1]*bc[1]
        mag = math.hypot(*ba) * math.hypot(*bc)
        return math.degrees(math.acos(dot/mag)) if mag else 0

    return {
        "knee_angle": angle(hip, knee, ankle),
        "trunk_angle": angle(shoulder, hip, knee),
    }

