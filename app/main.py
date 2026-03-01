# app/main.py
import json

from fastapi import FastAPI, WebSocket, Depends, HTTPException, WebSocketDisconnect
from sqlalchemy.orm import Session, joinedload
import os
from datetime import datetime, date, time
from .database import engine, Base, get_db,SessionLocal
from .session.session_manager import SessionManager, SessionMode
from .config import UPLOAD_DIR
from typing import List
from pydantic import BaseModel, Field
from .models import User, Attendance  
from starlette.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Base.metadata.create_all(bind=engine)

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# ── Registration WebSocket ────────────────────────────────────────────────
@app.websocket("/ws/register")
async def ws_register(websocket: WebSocket):
    await websocket.accept()
    db: Session = SessionLocal()

    print("[WS REGISTER] Connection accepted")

    try:
        print("[WS REGISTER] Waiting for first JSON message...")
        init_data = await websocket.receive_json()
        print(f"[WS REGISTER] Received first message: {init_data}")

        user_name = init_data.get("user_name")
        if not user_name or not isinstance(user_name, str) or len(user_name.strip()) < 3:
            print(f"[WS REGISTER] Invalid user_name: {user_name}")
            await websocket.send_text("Invalid user_name - must be string, min 3 chars")
            return

        print(f"[WS REGISTER] Starting session for user: {user_name}")
        manager = SessionManager(
            websocket=websocket,
            db=db,
            mode=SessionMode.REGISTER,
            user_name=user_name.strip()
        )

        await manager.run()

    except WebSocketDisconnect as e:
        print(f"[WS REGISTER] Client disconnected normally (code: {e.code})")

    except Exception as e:
        print(f"[WS REGISTER] Exception occurred: {type(e).__name__}: {str(e)}")
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_text(f"Server error: {str(e)}")

    finally:
        db.close()
        print(f"[WS REGISTER] Cleaning up connection")
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.close()
            print("[WS REGISTER] Closed by server")
        else:
            print("[WS REGISTER] Already closed or disconnecting")

# ── Recognition (Attendance) WebSocket ────────────────────────────────────
@app.websocket("/ws/recognize")
async def ws_recognize(websocket: WebSocket):
    await websocket.accept()
    db: Session = SessionLocal()

    try:
        print("[WS RECOGNIZE] Waiting for init JSON...")
        raw = await websocket.receive_text()   # ← receive as text first (safer)
        print(f"[WS RECOGNIZE] First message received: {raw!r}")

        try:
            init_data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[WS RECOGNIZE] Invalid JSON: {e}")
            await websocket.send_text("Invalid JSON format - send {\"mode\": \"recognize\"}")
            return

        if init_data.get("mode") != "recognize":
            await websocket.send_text("Expected {\"mode\": \"recognize\"}")
            return

        print("[WS RECOGNIZE] Init OK → starting session")
        manager = SessionManager(
            websocket=websocket,
            db=db,
            mode=SessionMode.RECOGNIZE,
            user_name=None
        )
        await manager.run()

    except WebSocketDisconnect:
        print("[WS RECOGNIZE] Client disconnected normally")
    except Exception as e:
        print(f"[WS RECOGNIZE] Error: {type(e).__name__}: {str(e)}")
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_text(f"Server error: {str(e)}")
    finally:
        db.close()
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.close()
            
class UserOut(BaseModel):
    id: int
    name: str
    created_at: datetime | None
    embeddings_count: int

    class Config:
        from_attributes = True


class AttendanceOut(BaseModel):
    attendance_id: int
    user_id: int
    name: str
    timestamp: datetime

    class Config:
        from_attributes = True


class AttendanceSimple(BaseModel):
    date: date
    timestamp: datetime


class UserDetailOut(BaseModel):
    id: int
    name: str
    created_at: datetime | None
    embeddings_count: int
    recent_attendance: List[AttendanceSimple] = Field(default_factory=list)

    class Config:
        from_attributes = True

@app.get("/users/", response_model=List[UserOut])
def get_users():
    db: Session = SessionLocal()
    try:
        users = db.query(User).options(joinedload(User.embeddings)).all()

        return [
            UserOut(
                id=u.id,
                name=u.name,
                created_at=u.created_at,
                embeddings_count=len(u.embeddings)
            )
            for u in users
        ]
    finally:
        db.close()

@app.get("/users/{user_id}", response_model=UserDetailOut)
def get_user_detail(user_id: int):
    db: Session = SessionLocal()

    try:
        user = (
            db.query(User)
            .options(joinedload(User.embeddings))
            .filter(User.id == user_id)
            .first()
        )

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        recent = (
            db.query(Attendance)
            .filter(Attendance.user_id == user_id)
            .order_by(Attendance.timestamp.desc())
            .limit(10)
            .all()
        )

        return UserDetailOut(
            id=user.id,
            name=user.name,
            created_at=user.created_at,
            embeddings_count=len(user.embeddings),
            recent_attendance=[
                AttendanceSimple(
                    date=a.timestamp.date(),
                    timestamp=a.timestamp
                )
                for a in recent
            ]
        )
    finally:
        db.close()


@app.get("/attendance/today", response_model=List[AttendanceOut])
def get_today_attendance():
    db: Session = SessionLocal()

    try:
        today = date.today()
        start = datetime.combine(today, time.min)
        end = datetime.combine(today, time.max)

        records = (
            db.query(Attendance)
            .join(User)
            .options(joinedload(Attendance.user))
            .filter(Attendance.timestamp >= start)
            .filter(Attendance.timestamp <= end)
            .order_by(Attendance.timestamp.desc())
            .all()
        )

        return [
            AttendanceOut(
                attendance_id=a.id,
                user_id=a.user_id,
                name=a.user.name,
                timestamp=a.timestamp
            )
            for a in records
        ]
    finally:
        db.close()