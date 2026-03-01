# app/face/landmark_service.py
import mediapipe as mp
import cv2
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

def detect_landmarks(frame: np.ndarray):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape
    return {
        "nose": np.array([lm[1].x * w, lm[1].y * h, lm[1].z]),
        "left_eye": np.array([lm[33].x * w, lm[33].y * h, lm[33].z]),
        "right_eye": np.array([lm[263].x * w, lm[263].y * h, lm[263].z]),
        "mouth_left": np.array([lm[61].x * w, lm[61].y * h]),
        "mouth_right": np.array([lm[291].x * w, lm[291].y * h]),
        "mouth_top": np.array([lm[13].x * w, lm[13].y * h]),
        "mouth_bottom": np.array([lm[14].x * w, lm[14].y * h]),
        "left_eye_top": np.array([lm[159].x * w, lm[159].y * h]),
        "left_eye_bottom": np.array([lm[145].x * w, lm[145].y * h]),
        "right_eye_top": np.array([lm[386].x * w, lm[386].y * h]),
        "right_eye_bottom": np.array([lm[374].x * w, lm[374].y * h]),
    }