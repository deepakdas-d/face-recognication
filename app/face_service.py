from deepface import DeepFace
import numpy as np
import cv2
import os
from app.config import MODEL_NAME, DETECTOR_BACKEND
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognitionError(Exception):
    pass

def extract_embedding(image_path):
    """
    Extract face embedding from image
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            raise FaceRecognitionError(f"Image file not found: {image_path}")
        
        # Check if image is valid
        img = cv2.imread(image_path)
        if img is None:
            raise FaceRecognitionError("Invalid image file")
        
        # Extract embedding
        result = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False  # Don't enforce face detection to avoid errors
        )
        
        if not result:
            raise FaceRecognitionError("No face detected in image")
        
        logger.info(f"Successfully extracted embedding from {image_path}")
        return result[0]["embedding"]
        
    except Exception as e:
        logger.error(f"Error extracting embedding: {str(e)}")
        raise FaceRecognitionError(f"Failed to extract embedding: {str(e)}")

def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors
    """
    try:
        a = np.array(a, dtype=np.float64)
        b = np.array(b, dtype=np.float64)
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return float(np.dot(a, b) / (norm_a * norm_b))
        
    except Exception as e:
        logger.error(f"Error calculating similarity: {str(e)}")
        return 0.0

def validate_embedding(embedding):
    """
    Validate that embedding is properly formatted
    """
    try:
        if not isinstance(embedding, (list, np.ndarray)):
            return False
        if len(embedding) == 0:
            return False
        return True
    except:
        return False