import cv2
import numpy as np
import logging
from deepface import DeepFace
from typing import Optional
from ..config import MODEL_NAME, DETECTOR_BACKEND, THRESHOLD

# ───────────────────────────────
# Logger setup
# ───────────────────────────────
logger = logging.getLogger(__name__)
# Only configure if not already configured (prevents duplicate handlers in reloads)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
logger.info(f"[EMBEDDING INIT] MODEL_NAME={MODEL_NAME} | DETECTOR_BACKEND={DETECTOR_BACKEND}")

# ───────────────────────────────
# Embedding extraction
# ───────────────────────────────
def extract_embedding(frame: np.ndarray) -> Optional[np.ndarray]:
    if frame is None or frame.size == 0:
        logger.warning("[EMBEDDING] empty frame received")
        return None

    if frame.dtype != np.uint8:
        logger.error(f"[EMBEDDING] invalid dtype: {frame.dtype}")
        return None

    if len(frame.shape) != 3 or frame.shape[2] != 3:
        logger.error(f"[EMBEDDING] invalid shape: {frame.shape}")
        return None

    # Explicitly ensure it's a contiguous numpy array in RGB
    frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    h, w = frame.shape[:2]
    if h < 64 or w < 64:  # retinaface prefers slightly larger min size
        logger.warning(f"[EMBEDDING] image too small for retinaface: {frame.shape}")
        return None

    try:
        embeddings = DeepFace.represent(
            img_path=frame,                    # now guaranteed RGB np.array
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
            normalization="ArcFace"
        )

        if not embeddings or len(embeddings) == 0:
            logger.warning("[EMBEDDING] No faces or embeddings returned")
            return None

        emb_dict = embeddings[0]
        emb = emb_dict.get("embedding")
        if emb is None:
            return None

        emb_array = np.array(emb, dtype=np.float32)
        if emb_array.shape != (512,):
            logger.warning(f"[EMBEDDING] unexpected embedding size: {emb_array.shape}")
            return None

        # Face size from detection (retinaface provides this)
        facial_area = emb_dict.get("facial_area", {})
        w_area = facial_area.get("w", 0)
        h_area = facial_area.get("h", 0)
        if w_area < 80 or h_area < 80:
            logger.warning(f"[EMBEDDING] Detected face too small: {w_area}x{h_area}")
            return None

        return emb_array

    except Exception as e:
        logger.exception(f"[EMBEDDING] DeepFace failed - detector={DETECTOR_BACKEND}")
        return None

# ───────────────────────────────
# Embedding comparison
# ───────────────────────────────
def compare_embeddings(embedding1: np.ndarray, embedding2: np.ndarray) -> bool:
    if embedding1 is None or embedding2 is None:
        logger.warning("[COMPARE] one or both embeddings are None")
        return False

    try:
        emb1 = embedding1.astype(np.float32)
        emb2 = embedding2.astype(np.float32)

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            logger.warning("[COMPARE] zero or near-zero norm detected")
            return False

        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        return similarity > THRESHOLD

    except Exception as e:
        logger.error(f"[COMPARE] error: {str(e)}")
        return False


# ───────────────────────────────
# Cosine similarity utility
# ───────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    try:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    except Exception as e:
        logger.error(f"[COSINE] error: {str(e)}")
        return 0.0