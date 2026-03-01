# app/attendance_service.py
from datetime import datetime
from sqlalchemy.orm import Session
from .models import Attendance

def mark_attendance(db: Session, user_id: int):
    attendance = Attendance(user_id=user_id, timestamp=datetime.utcnow())
    db.add(attendance)
    db.commit()