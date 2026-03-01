from datetime import date
from sqlalchemy.orm import Session
from app.models import Attendance
import logging

logger = logging.getLogger(__name__)

def mark_attendance(db: Session, user_id: int):
    """
    Mark attendance for a user if not already marked today
    """
    try:
        # Check if attendance already marked today
        existing = db.query(Attendance).filter(
            Attendance.user_id == user_id,
            Attendance.date == date.today()
        ).first()

        if existing:
            logger.info(f"Attendance already marked for user {user_id} today")
            return "Already Marked"

        # Mark new attendance
        attendance = Attendance(user_id=user_id)
        db.add(attendance)
        db.commit()
        
        logger.info(f"Attendance marked for user {user_id}")
        return "Marked"
        
    except Exception as e:
        logger.error(f"Error marking attendance: {str(e)}")
        db.rollback()
        raise

def get_today_attendance(db: Session):
    """
    Get all attendance records for today
    """
    try:
        attendance_records = db.query(Attendance).filter(
            Attendance.date == date.today()
        ).all()
        
        return attendance_records
        
    except Exception as e:
        logger.error(f"Error fetching attendance: {str(e)}")
        return []