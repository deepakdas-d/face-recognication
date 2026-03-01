import os
from dotenv import load_dotenv

load_dotenv()

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./face.db")

# Recognition settings
THRESHOLD = float(os.getenv("THRESHOLD", 0.6))
MODEL_NAME = os.getenv("MODEL_NAME", "ArcFace")
DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "opencv")

# Upload settings
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 5 * 1024 * 1024))  # 5MB default

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)