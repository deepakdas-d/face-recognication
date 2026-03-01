# app/face/pose_service.py
import numpy as np

def estimate_pose(coords):
    if not coords:
        return None

    nose = coords["nose"]
    left_eye = coords["left_eye"]
    right_eye = coords["right_eye"]
    eye_center = (left_eye + right_eye) / 2
    nose_vec = nose - eye_center

    yaw = np.arctan2(nose_vec[0], nose_vec[2])
    pitch = np.arctan2(nose_vec[1], nose_vec[2])
    roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])

    # Mouth metrics
    mouth_width = np.linalg.norm(coords["mouth_right"] - coords["mouth_left"])
    mouth_height = np.linalg.norm(coords["mouth_bottom"] - coords["mouth_top"])

    # Eye metrics
    left_eye_height = np.linalg.norm(coords["left_eye_top"] - coords["left_eye_bottom"])
    right_eye_height = np.linalg.norm(coords["right_eye_top"] - coords["right_eye_bottom"])

    # Normalize by interocular distance
    interocular_dist = np.linalg.norm(right_eye[:2] - left_eye[:2])
    if interocular_dist > 0:
        mouth_width /= interocular_dist
        mouth_height /= interocular_dist
        left_eye_height /= interocular_dist
        right_eye_height /= interocular_dist

    return {
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "mouth_width": mouth_width,
        "mouth_height": mouth_height,
        "left_eye_height": left_eye_height,
        "right_eye_height": right_eye_height,
    }