# app/face/face_service.py
import numpy as np
import logging
from deepface import DeepFace
from typing import Optional
from app.config import MODEL_NAME, DETECTOR_BACKEND, THRESHOLD

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class FaceRecognitionError(Exception):
    pass

def extract_embedding_from_frame(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract face embedding from a live frame (NumPy array) using DeepFace.
    Returns 512-dim embedding or None if extraction fails.
    """
    if frame is None or frame.size == 0:
        logger.warning("[EMBEDDING] empty frame received")
        return None

    try:
        # DeepFace works with in-memory arrays for retinaface/mediapipe backends
        embedding_objs = DeepFace.represent(
            img_path=frame,  # NumPy array
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,  # should be retinaface, mediapipe, or yunet
            enforce_detection=True,
            align=True,
            normalization="base"
        )

        if not embedding_objs or len(embedding_objs) == 0:
            logger.debug("[EMBEDDING] no face detected")
            return None

        embedding = embedding_objs[0].get("embedding", None)
        if embedding is None:
            logger.debug("[EMBEDDING] embedding missing in DeepFace result")
            return None

        emb_array = np.array(embedding, dtype=np.float32)
        if emb_array.ndim != 1 or emb_array.shape[0] == 0:
            logger.error(f"[EMBEDDING] invalid embedding shape: {emb_array.shape}")
            return None

        logger.debug(f"[EMBEDDING] successfully extracted embedding (shape={emb_array.shape})")
        return emb_array

    except Exception as e:
        logger.error(f"[EMBEDDING] failed to extract: {str(e)}")
        return None

def compare_embeddings(embedding1: np.ndarray, embedding2: np.ndarray) -> bool:
    """
    Compare two embeddings using cosine similarity.
    Returns True if similarity > THRESHOLD.
    """
    if embedding1 is None or embedding2 is None:
        return False

    try:
        emb1 = embedding1.astype(np.float32)
        emb2 = embedding2.astype(np.float32)

        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return False

        similarity = dot / (norm1 * norm2)
        return similarity > THRESHOLD

    except Exception as e:
        logger.error(f"[COMPARE] error comparing embeddings: {str(e)}")
        return False
    
def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    try:
        a = np.array(a, dtype=np.float64)
        b = np.array(b, dtype=np.float64)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    except Exception as e:
        logger.error(f"[COSINE] error: {str(e)}")
        return 0.0