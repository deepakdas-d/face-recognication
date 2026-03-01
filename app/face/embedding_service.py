# app/face/embedding_service.py
from deepface import DeepFace
import numpy as np
from ..config import MODEL_NAME, DETECTOR_BACKEND, THRESHOLD

def extract_embedding(frame):
    try:
        embedding_objs = DeepFace.represent(frame, model_name=MODEL_NAME, detector_backend=DETECTOR_BACKEND, enforce_detection=False)
        if embedding_objs:
            return np.array(embedding_objs[0]["embedding"])
    except Exception as e:
        print(f"Embedding extraction error: {e}")
    return None

def compare_embeddings(embedding1, embedding2):
    # Cosine similarity
    dot = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    similarity = dot / (norm1 * norm2)
    return similarity > THRESHOLD