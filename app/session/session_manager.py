# app/session/session_manager.py
import asyncio
from enum import Enum
import time
import base64
import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from ..face.landmark_service import detect_landmarks
from ..face.pose_service import estimate_pose
from ..face.liveness_service import LivenessValidator
from ..face.embedding_service import extract_embedding, compare_embeddings
from ..face.anti_spoof_service import detect_spoof
from .challenge_generator import generate_challenge
from ..config import SESSION_TIMEOUT, MAX_EMBEDDINGS_PER_USER
from ..models import User, Embedding
from ..attendance_service import mark_attendance

class SessionState(Enum):
    INIT = 1
    FACE_DETECTED = 2
    CHALLENGE_SENT = 3
    MOVEMENT_VALIDATING = 4
    LIVENESS_PASSED = 5
    EMBEDDING_PROCESS = 6
    DONE = 7


class SessionMode(Enum):
    RECOGNIZE = "recognize"
    REGISTER = "register"


class SessionManager:
    def __init__(
        self,
        websocket: WebSocket,
        db: Session,
        mode: SessionMode,
        user_name: str | None = None
    ):
        self.websocket = websocket
        self.db = db
        self.mode = mode
        self.user_name = user_name
        self.state = SessionState.INIT
        self.challenge = None
        self.validator = None
        self.embedding = None
        self.start_time = time.time()
        self.frame_count = 0

    async def process_frame(self, frame_data: str) -> bool:
        self.frame_count += 1
        print(f"[FRAME {self.frame_count:03d}] size={len(frame_data):,} bytes")

        try:
            frame_bytes = base64.b64decode(frame_data)
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                print("[FRAME] Decode failed")
                await self.websocket.send_text("Invalid frame data")
                return False

            print(f"[FRAME {self.frame_count:03d}] decoded shape={frame.shape}")
        except Exception as e:
            print(f"[FRAME] Decode exception: {type(e).__name__}: {e}")
            await self.websocket.send_text("Frame decode error")
            return False

        if time.time() - self.start_time > SESSION_TIMEOUT:
            print("[SESSION] Timeout reached")
            await self.websocket.send_text("Session timed out")
            return False

        landmarks = detect_landmarks(frame)
        has_face = bool(landmarks)
        print(f"[FRAME {self.frame_count:03d}] face detected: {has_face}")

        pose = estimate_pose(landmarks) if has_face else None
        if pose:
            print(f"[POSE] yaw:{pose['yaw']:+.3f} pitch:{pose['pitch']:+.3f} roll:{pose['roll']:+.3f}")

        is_real = True  # spoof skipped
        print(f"[SPOOF] skipped → considered real")

        # ── State machine ───────────────────────────────────────
        if self.state == SessionState.INIT:
            if has_face:
                self.state = SessionState.FACE_DETECTED
                self.challenge = generate_challenge()
                await self.websocket.send_text(f"Challenge: {self.challenge}")
                print(f"[STATE] → FACE_DETECTED | Challenge sent: {self.challenge}")
                self.validator = LivenessValidator(self.challenge)
                self.state = SessionState.CHALLENGE_SENT

        elif self.state in (SessionState.CHALLENGE_SENT, SessionState.MOVEMENT_VALIDATING):
            if not has_face:
                print("[LIVENESS] No face → waiting")
            elif self.validator.validate_movement(pose, landmarks):
                self.state = SessionState.LIVENESS_PASSED
                await self.websocket.send_text("Liveness passed. Processing...")
                print("[STATE] → LIVENESS_PASSED")
            else:
                self.state = SessionState.MOVEMENT_VALIDATING

        elif self.state == SessionState.LIVENESS_PASSED:
            embedding = extract_embedding(frame)
            if embedding is not None:
                self.embedding = embedding
                self.state = SessionState.EMBEDDING_PROCESS
                print("[EMBEDDING] extracted successfully")
            else:
                print("[EMBEDDING] failed to extract")

        elif self.state == SessionState.EMBEDDING_PROCESS:
            # ── Final actions ─────────────────────────────────
            if self.mode == SessionMode.RECOGNIZE:
                user = self._recognize()
                if user:
                    mark_attendance(self.db, user.id)
                    await self.websocket.send_text(f"Recognized: {user.name}")
                    print(f"[RECOGNIZE] Success → {user.name}")
                else:
                    await self.websocket.send_text("Not recognized")
                    print("[RECOGNIZE] No match")
            elif self.mode == SessionMode.REGISTER:
                try:
                    self._register()
                    await self.websocket.send_text(f"Registered: {self.user_name}")
                    print(f"[REGISTER] Success → {self.user_name}")
                except ValueError as e:
                    await self.websocket.send_text(str(e))
                    print(f"[REGISTER] Failed: {e}")

            self.state = SessionState.DONE
            return False  # stop loop

        return True
    def _recognize(self) -> User | None:
        users = self.db.query(User).all()
        for user in users:
            for emb in user.embeddings:
                if compare_embeddings(self.embedding, np.array(emb.embedding)):
                    return user
        return None

    def _register(self):
        if self.mode != SessionMode.REGISTER or not self.user_name:
            raise ValueError("Invalid registration context")

        existing = self.db.query(User).filter(User.name == self.user_name).first()
        if existing:
            raise ValueError(f"User '{self.user_name}' already exists")

        user = User(name=self.user_name)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)

        # Add new embedding (replace oldest if limit reached)
        if len(user.embeddings) >= MAX_EMBEDDINGS_PER_USER:
            oldest = min(user.embeddings, key=lambda e: e.id)
            self.db.delete(oldest)
            self.db.commit()

        new_emb = Embedding(
            embedding=self.embedding.tolist(),
            user_id=user.id
        )
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
            # Client closed connection: stop silently
            pass
        except Exception as e:
            try:
                await self.websocket.send_text(f"Error: {str(e)}")
            except:
                # WebSocket already closed
                pass
        finally:
            try:
                await self.websocket.close()
            except:
                # Socket may already be closed
                pass