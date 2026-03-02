import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./face.db")
THRESHOLD = float(os.getenv("THRESHOLD", 0.6))
MODEL_NAME = os.getenv("MODEL_NAME", "ArcFace")
DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "mediapipe")   # ← changed default
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 5242880))
SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", 30))
ANTI_SPOOF_THRESHOLD = float(os.getenv("ANTI_SPOOF_THRESHOLD", 0.65))
MAX_EMBEDDINGS_PER_USER = int(os.getenv("MAX_EMBEDDINGS_PER_USER", 5))