# app/face/liveness_service.py
from collections import deque
import numpy as np
import time

class LivenessValidator:
    def __init__(self, challenge, max_frames=60, min_frames_for_check=12):
        self.challenge = challenge
        self.pose_history = deque(maxlen=max_frames)
        self.validated = False
        self.min_frames_for_check = min_frames_for_check
        self.start_time = time.time()

    def average_pose(self, poses):
        if not poses:
            return None
        return {
            "yaw":            np.mean([p["yaw"]            for p in poses]),
            "pitch":          np.mean([p["pitch"]          for p in poses]),
            "roll":           np.mean([p["roll"]           for p in poses]),
            "mouth_width":    np.mean([p["mouth_width"]    for p in poses]),
            "mouth_height":   np.mean([p["mouth_height"]   for p in poses]),
            "left_eye_height": np.mean([p["left_eye_height"] for p in poses]),
            "right_eye_height": np.mean([p["right_eye_height"] for p in poses]),
        }

    def validate_movement(self, pose, landmarks=None):
        if pose is None:
            print("[LIVENESS] Pose is None → skipping")
            return False

        self.pose_history.append({"pose": pose, "landmarks": landmarks, "ts": time.time()})

        history_len = len(self.pose_history)
        print(f"[LIVENESS] History size: {history_len} | Challenge: {self.challenge}")

        if history_len < self.min_frames_for_check:
            print(f"[LIVENESS] Not enough frames yet ({history_len}/{self.min_frames_for_check})")
            return False

        # Use last 8 frames for recent average (smoother)
        recent_poses = [f["pose"] for f in list(self.pose_history)[-8:]]
        recent_pose = self.average_pose(recent_poses)

        initial_pose = self.pose_history[0]["pose"]
        print(f"[DEBUG POSE] Initial: yaw={initial_pose['yaw']:.4f} pitch={initial_pose['pitch']:.4f}")
        print(f"[DEBUG POSE] Recent avg: yaw={recent_pose['yaw']:.4f} pitch={recent_pose['pitch']:.4f}")

        delta_yaw   = recent_pose["yaw"]   - initial_pose["yaw"]
        delta_pitch = recent_pose["pitch"] - initial_pose["pitch"]
        delta_roll  = recent_pose["roll"]  - initial_pose["roll"]
        print(f"[DEBUG DELTA] Δyaw={delta_yaw:+.4f}  Δpitch={delta_pitch:+.4f}  Δroll={delta_roll:+.4f}")

        print(f"[LIVENESS] Δyaw: {delta_yaw:+.4f}  Δpitch: {delta_pitch:+.4f}  Δroll: {delta_roll:+.4f}")

        validated = False

        if self.challenge == "turn_left":
            validated = delta_yaw < -0.8       # user needs to turn clearly left → more negative yaw

        elif self.challenge == "turn_right":
            validated = delta_yaw > +0.8
        elif self.challenge == "look_up":
            validated = delta_pitch < -0.05  # Negative delta = up (nose higher)
        elif self.challenge == "look_down":
            validated = delta_pitch > 0.05   # Positive delta = down (nose lower)
        elif self.challenge == "smile":
            validated = self._detect_smile(recent_poses)
        elif self.challenge == "blink":
            validated = self._detect_blink(recent_poses)
        if validated:
            print(f"[LIVENESS] VALIDATED! Challenge: {self.challenge}")
            self.validated = True
        else:
            print(f"[LIVENESS] Not yet satisfied")

        return validated

    def _detect_smile(self, recent_poses):
        ratios = [p["mouth_width"] / max(p["mouth_height"], 0.001) for p in recent_poses]
        max_ratio = max(ratios)
        print(f"[SMILE] max mouth width/height ratio: {max_ratio:.3f}")
        return max_ratio > 1.45   # lowered a bit — tune after seeing real values

    def _detect_blink(self, recent_poses):
        ears = [(p["left_eye_height"] + p["right_eye_height"]) / 2 for p in recent_poses]
        min_ear = min(ears)
        print(f"[BLINK] min EAR: {min_ear:.4f}")
        return min_ear < 0.22   # tune based on your camera / lighting