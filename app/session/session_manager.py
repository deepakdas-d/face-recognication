# app/session/session_manager.py
import asyncio
from enum import Enum
import time
import base64
import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from ..face.embedding_service import extract_embedding, compare_embeddings, cosine_similarity
from ..face.landmark_service import detect_landmarks
from ..face.pose_service import estimate_pose
from ..face.liveness_service import LivenessValidator
from ..face.anti_spoof_service import detect_spoof
from .challenge_generator import generate_challenge
from ..config import SESSION_TIMEOUT, MAX_EMBEDDINGS_PER_USER, THRESHOLD
from ..models import User, Embedding
from ..attendance_service import mark_attendance

class SessionState(Enum):
    INIT = 1
    FACE_DETECTED = 2
    CHALLENGE_SENT = 3
    LIVENESS_VALIDATING = 4
    LIVENESS_PASSED = 5
    EMBEDDING_ACCUMULATING = 6
    DONE = 7

class SessionMode(Enum):
    RECOGNIZE = "recognize"
    REGISTER = "register"

class SessionManager:
    def __init__(self, websocket: WebSocket, db: Session, mode: SessionMode, user_name: str | None = None):
        self.websocket = websocket
        self.db = db
        self.mode = mode
        self.user_name = user_name
        self.state = SessionState.INIT
        self.challenge = None
        self.validator = None
        self.start_time = time.time()
        self.frame_count = 0
        self.valid_embeddings = []  # accumulate embeddings over frames
        self.pose_sequence = None
        self.current_pose_idx = 0
        self.collected_per_pose = 0
        self.pose_start_time = None  # new: track time per pose to avoid infinite stuck

    async def process_frame(self, frame_data: str) -> bool:
        self.frame_count += 1
        print(f"[FRAME {self.frame_count:03d}] size={len(frame_data):,} bytes")

        # Decode frame
        try:
            frame_bytes = base64.b64decode(frame_data)
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                print("[FRAME] Decode failed")
                await self.websocket.send_text("Invalid frame data")
                return True  # skip frame
            print(f"[FRAME {self.frame_count:03d}] decoded shape={frame.shape}")
        except Exception as e:
            print(f"[FRAME] Decode exception: {type(e).__name__}: {e}")
            await self.websocket.send_text("Frame decode error")
            return True  # skip frame

        # Timeout check (global session)
        if time.time() - self.start_time > SESSION_TIMEOUT:
            print("[SESSION] Timeout reached")
            await self.websocket.send_text("Session timed out")
            return False

        # Landmark detection & pose estimation
        landmarks = detect_landmarks(frame)
        has_face = bool(landmarks)
        print(f"[FRAME {self.frame_count:03d}] face detected: {has_face}")

        pose = estimate_pose(landmarks) if has_face else None
        if pose:
            print(f"[POSE] yaw:{pose['yaw']:+.3f} pitch:{pose['pitch']:+.3f} roll:{pose['roll']:+.3f}")

        # Skip spoof for now
        is_real = True
        print(f"[SPOOF] skipped → considered real")

        # ── State Machine ─────────────────────────────────────────────
        if self.state == SessionState.INIT and has_face:
            self.challenge = generate_challenge()
            self.validator = LivenessValidator(self.challenge)
            await self.websocket.send_text(f"Challenge: {self.challenge}")
            print(f"[STATE] → FACE_DETECTED | Challenge sent: {self.challenge}")
            self.state = SessionState.CHALLENGE_SENT

        elif self.state in (SessionState.CHALLENGE_SENT, SessionState.LIVENESS_VALIDATING):
            if has_face and self.validator.validate_movement(pose, landmarks):
                self.state = SessionState.LIVENESS_PASSED
                await self.websocket.send_text("Liveness passed. Collecting embeddings...")
                print("[STATE] → LIVENESS_PASSED")
                if self.mode == SessionMode.REGISTER:
                    # Initialize pose sequence for registration
                    self.pose_sequence = [
                        {"name": "frontal", 
                         "msg": "Please face the camera directly. Capturing in 2 seconds...", 
                         "yaw_range": (-20, 20)},

                        # {"name": "left", 
                        #  "msg": "Now slowly turn your head LEFT (look toward your left shoulder). Capturing in 2 seconds...", 
                        #  "yaw_range": (-15, -4)},   # widened for easier trigger / debugging

                        # {"name": "right", 
                        #  "msg": "Now slowly turn your head RIGHT (look toward your right shoulder). Capturing in 2 seconds...", 
                        #  "yaw_range": (4,15)},     # symmetric, widened
                    ]
                    self.current_pose_idx = 0
                    self.collected_per_pose = 0
                    self.pose_start_time = time.time()  # start timer for first pose
                    await self.websocket.send_text(self.pose_sequence[0]["msg"])
                    await asyncio.sleep(2)  # initial delay
            else:
                self.state = SessionState.LIVENESS_VALIDATING

        elif self.state in (SessionState.LIVENESS_PASSED, SessionState.EMBEDDING_ACCUMULATING):
            if has_face:
                # Quality checks: blur and face size
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if lap_var < 100:
                    print(f"[QUALITY] Skipping blurry frame (variance={lap_var:.2f})")
                    await self.websocket.send_text("Frame too blurry – please hold still")
                    return True

                emb = extract_embedding(frame)
                if emb is None:
                    print(f"[EMBEDDING] failed for frame {self.frame_count:03d}")
                    await self.websocket.send_text("Face not clear – please stay in view")
                    return True

                if self.mode == SessionMode.RECOGNIZE:
                    self.valid_embeddings.append(emb)
                    if len(self.valid_embeddings) > 10:
                        self.valid_embeddings.pop(0)
                    count = len(self.valid_embeddings)
                    print(f"[EMBEDDING] collected {count} / 5 required (frame {self.frame_count:03d})")
                    await self.websocket.send_text(f"Collecting face data... {count}/5 embeddings captured")
                    self.state = SessionState.EMBEDDING_ACCUMULATING
                    if count >= 5:
                        avg_embedding = np.mean(np.stack(self.valid_embeddings), axis=0)
                        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
                        user = self._recognize(avg_embedding)
                        if user:
                            mark_attendance(self.db, user.id)
                            await self.websocket.send_text(f"Recognized: {user.name}")
                            print(f"[RECOGNIZE] Success → {user.name}")
                        else:
                            await self.websocket.send_text("Not recognized")
                            print("[RECOGNIZE] No match")
                        self.state = SessionState.DONE
                        return False

                elif self.mode == SessionMode.REGISTER:
                    if self.pose_sequence is None:
                        return True

                    current_pose = self.pose_sequence[self.current_pose_idx]
                    yaw = pose['yaw']
                    min_yaw, max_yaw = current_pose["yaw_range"]

                    # Reset per-pose timer on new pose
                    if self.current_pose_idx > 0 and self.collected_per_pose == 0:
                        self.pose_start_time = time.time()

                    # Per-pose timeout (e.g. 30 seconds) - prevent infinite stuck
                    if time.time() - self.pose_start_time > 30:
                        print(f"[REG-TIMEOUT] Pose '{current_pose['name']}' timed out after 30s")
                        self.current_pose_idx += 1
                        self.collected_per_pose = 0
                        if self.current_pose_idx < len(self.pose_sequence):
                            next_pose = self.pose_sequence[self.current_pose_idx]
                            await self.websocket.send_text(next_pose["msg"])
                            await asyncio.sleep(2)
                            self.pose_start_time = time.time()
                        else:
                            # Force finish with what we have
                            try:
                                for emb_item in self.valid_embeddings:
                                    self._register_single(emb_item)
                                await self.websocket.send_text(f"Registered: {self.user_name} (partial views)")
                                print(f"[REGISTER] Partial success → {self.user_name}")
                            except ValueError as e:
                                await self.websocket.send_text(f"Registration failed: {str(e)}")
                            self.state = SessionState.DONE
                            return False

                    # ── DEBUG LOG FOR LEFT/RIGHT MOVEMENT ────────────────────────────────
                    direction_str = "LEFT" if current_pose['name'] == "left" else \
                                    "RIGHT" if current_pose['name'] == "right" else "FRONTAL"
                    
                    status = "ACCEPT" if min_yaw <= yaw <= max_yaw else "REJECT"
                    
                    print(f"[REG-DEBUG] Pose #{self.current_pose_idx} '{current_pose['name']}' | "
                          f"yaw = {yaw:+.1f}° | expected [{min_yaw}, {max_yaw}] | {status} | "
                          f"collected so far: {self.collected_per_pose}/1")  # note: /1 for now
                    
                    if abs(yaw) > 5:
                        turn_dir = "LEFT" if yaw < 0 else "RIGHT"
                        print(f"          → Head is turned {turn_dir} (yaw sign: {'negative' if yaw < 0 else 'positive'})")
                    # ──────────────────────────────────────────────────────────────────────

                    if min_yaw <= yaw <= max_yaw:
                        self.valid_embeddings.append(emb)
                        self.collected_per_pose += 1
                        print(f"[POSE {current_pose['name']}] collected {self.collected_per_pose}/1 (good yaw={yaw:.1f})")
                        await self.websocket.send_text(f"Captured for {current_pose['name']} ({self.collected_per_pose}/1)")
                        
                        if self.collected_per_pose >= 1:  # ← temporarily 1 instead of 2
                            self.current_pose_idx += 1
                            self.collected_per_pose = 0
                            if self.current_pose_idx < len(self.pose_sequence):
                                next_pose = self.pose_sequence[self.current_pose_idx]
                                await self.websocket.send_text(next_pose["msg"])
                                await asyncio.sleep(2)
                                self.pose_start_time = time.time()
                            else:
                                # All poses done → register
                                try:
                                    for emb_item in self.valid_embeddings:
                                        self._register_single(emb_item)
                                    await self.websocket.send_text(f"Registered: {self.user_name} with multiple views!")
                                    print(f"[REGISTER] Success → {self.user_name}")
                                except ValueError as e:
                                    await self.websocket.send_text(f"Registration failed: {str(e)}")
                                    print(f"[REGISTER] Failed: {e}")
                                self.state = SessionState.DONE
                                return False
                    else:
                        direction = "LEFT" if current_pose['name'] == "left" else \
                                    "RIGHT" if current_pose['name'] == "right" else "straight at the camera"
                        msg = f"Head position not ideal for {current_pose['name']}. Please turn your head more to the {direction} side and hold still."

                        # Helpful guidance
                        if current_pose['name'] == "left":
                            if yaw >= 0:
                                msg += " (You're facing straight or slightly RIGHT – turn the other way!)"
                            elif yaw > min_yaw + 5:
                                msg += f" (Not far enough – current left turn is only ~{abs(yaw):.0f}°)"
                        elif current_pose['name'] == "right":
                            if yaw <= 0:
                                msg += " (You're facing straight or slightly LEFT – turn the other way!)"
                            elif yaw < max_yaw - 5:
                                msg += f" (Not far enough – current right turn is only ~{yaw:.0f}°)"

                        await self.websocket.send_text(msg)
                        # Retry same pose

        return True
    def _recognize(self, probe_embedding: np.ndarray) -> User | None:
        users = self.db.query(User).all()
        best_user = None
        best_score = -1.0
        
        for user in users:
            for stored_emb in user.embeddings:
                stored_array = np.array(stored_emb.embedding)
                score = cosine_similarity(probe_embedding, stored_array)
                print(f"[DEBUG SCORE] {user.name} → {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_user = user

        print(f"[RECOGNIZE] Best score overall: {best_score:.4f}")
        
        if best_user and best_score >= THRESHOLD:
            print(f"[RECOGNIZE] Matched {best_user.name} with score {best_score:.4f}")
            return best_user
        else:
            print(f"[RECOGNIZE] No match (best was {best_score:.4f} < {THRESHOLD})")
            return None

    def _register(self, emb: np.ndarray):
        # Legacy single register – not used now
        pass

    def _register_single(self, emb: np.ndarray):
        if self.mode != SessionMode.REGISTER or not self.user_name:
            raise ValueError("Invalid registration context")

        user = self.db.query(User).filter(User.name == self.user_name).first()
        if not user:
            # Create user if first embedding
            existing = self.db.query(User).filter(User.name == self.user_name).first()
            if existing:
                raise ValueError(f"User '{self.user_name}' already exists")
            user = User(name=self.user_name)
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)

        # Maintain max embeddings per user
        if len(user.embeddings) >= MAX_EMBEDDINGS_PER_USER:
            oldest = min(user.embeddings, key=lambda e: e.id)
            self.db.delete(oldest)
            self.db.commit()

        new_emb = Embedding(embedding=emb.tolist(), user_id=user.id)
        emb = emb / np.linalg.norm(emb)
        self.db.add(new_emb)
        self.db.commit()

    async def run(self):
        try:
            while True:
                data = await self.websocket.receive_text()
                if data.startswith("frame:"):
                    keep_going = await self.process_frame(data[6:])
                    if not keep_going:
                        break
                elif data == "close":
                    break
                else:
                    await self.websocket.send_text("Unknown command")
        except WebSocketDisconnect:
            print("[WS] Disconnected")
        except Exception as e:
            try:
                await self.websocket.send_text(f"Error: {str(e)}")
            except:
                pass
        finally:
            try:
                await self.websocket.close()
            except:
                pass