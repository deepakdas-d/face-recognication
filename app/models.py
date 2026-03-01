from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Date, Text, Index
from datetime import datetime, date
from app.database import Base

class User(Base):
    __tablename__ = "users"
    
    __table_args__ = (
        Index('idx_user_name', 'name'),
    )

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    
    __table_args__ = (
        Index('idx_face_user', 'user_id'),
    )

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    embedding = Column(Text, nullable=False)  # JSON string of embedding
    created_at = Column(DateTime, default=datetime.utcnow)

class Attendance(Base):
    __tablename__ = "attendance"
    
    __table_args__ = (
        Index('idx_attendance_user_date', 'user_id', 'date'),
    )

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, default=date.today, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)