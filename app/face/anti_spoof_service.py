# app/face/anti_spoof_service.py
import cv2
import numpy as np
from ..config import ANTI_SPOOF_THRESHOLD

def detect_spoof(frame):
    # Simple texture-based anti-spoof (LBP variance or something basic)
    # For production, use a proper model, but here's a placeholder
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    variance = laplacian.var()
    score = variance / 1000.0  # Normalize
    return score > ANTI_SPOOF_THRESHOLD