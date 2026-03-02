import numpy as np

def estimate_pose(coords):
    if not coords:
        return None

    nose = coords["nose"]
    left_eye = coords["left_eye"]
    right_eye = coords["right_eye"]

    # Eye center
    eye_center = (left_eye + right_eye) / 2.0

    # Interocular distance (scale normalization)
    interocular_dist = np.linalg.norm(right_eye[:2] - left_eye[:2])
    if interocular_dist <= 1e-6:
        return None

    # -------- YAW (2D proxy) --------
    # Horizontal nose offset normalized by eye distance
    yaw_proxy = (nose[0] - eye_center[0]) / interocular_dist

    # Scale to degrees-like range
    yaw = yaw_proxy * 60.0   # tune 40–80 depending on sensitivity

    # -------- ROLL --------
    roll = np.degrees(
        np.arctan2(
            right_eye[1] - left_eye[1],
            right_eye[0] - left_eye[0]
        )
    )

    # -------- PITCH (2D proxy) --------
    pitch_proxy = (nose[1] - eye_center[1]) / interocular_dist
    pitch = pitch_proxy * 60.0

    # -------- Extra Metrics --------
    mouth_width = np.linalg.norm(coords["mouth_right"] - coords["mouth_left"]) / interocular_dist
    mouth_height = np.linalg.norm(coords["mouth_bottom"] - coords["mouth_top"]) / interocular_dist
    left_eye_height = np.linalg.norm(coords["left_eye_top"] - coords["left_eye_bottom"]) / interocular_dist
    right_eye_height = np.linalg.norm(coords["right_eye_top"] - coords["right_eye_bottom"]) / interocular_dist

    return {
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "mouth_width": mouth_width,
        "mouth_height": mouth_height,
        "left_eye_height": left_eye_height,
        "right_eye_height": right_eye_height,
    }