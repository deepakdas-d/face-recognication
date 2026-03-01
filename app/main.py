from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status,Form
from sqlalchemy.orm import Session
import shutil
import os
import json
import uuid
import logging
from datetime import date

from app.database import SessionLocal, engine, Base
from app import models
from app.face_service import extract_embedding, cosine_similarity, FaceRecognitionError
from app.attendance_service import mark_attendance, get_today_attendance
from app.config import THRESHOLD, UPLOAD_DIR, MAX_FILE_SIZE
from fastapi.middleware.cors import CORSMiddleware

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Face Recognition Attendance System",
    description="API for face recognition based attendance marking",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://10.214.184.185:5173",  # Add your network IP
        "http://192.168.1.x:5173", 
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_upload_file(upload_file: UploadFile) -> str:
    """
    Save uploaded file to temporary location
    """
    # Generate unique filename
    file_extension = os.path.splitext(upload_file.filename)[1]
    file_name = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, file_name)
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    # Check file size
    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE:
        os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {MAX_FILE_SIZE} bytes"
        )
    
    return file_path

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "database": "connected"}

# Register User
@app.post("/register/", status_code=status.HTTP_201_CREATED)
async def register(
    name: str = Form(...), 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    """
    Register a new user with face image
    """
    temp_path = None
    try:
        # Check if user already exists
        existing_user = db.query(models.User).filter(
            models.User.name == name
        ).first()
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User already exists"
            )
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )
        
        # Save uploaded file
        temp_path = save_upload_file(file)
        
        # Extract face embedding
        try:
            embedding = extract_embedding(temp_path)
        except FaceRecognitionError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        
        # Create user
        user = models.User(name=name)
        db.add(user)
        db.commit()
        db.refresh(user)
        
        # Save face embedding
        face_emb = models.FaceEmbedding(
            user_id=user.id,
            embedding=json.dumps(embedding)
        )
        db.add(face_emb)
        db.commit()
        
        logger.info(f"User {name} registered successfully with ID: {user.id}")
        
        return {
            "message": "User registered successfully",
            "user_id": user.id,
            "name": user.name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in registration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# Recognize face and mark attendance
@app.post("/recognize/")
async def recognize(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    """
    Recognize face from image and mark attendance if recognized
    """
    temp_path = None
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )
        
        # Save uploaded file
        temp_path = save_upload_file(file)
        
        # Extract embedding from new image
        try:
            new_embedding = extract_embedding(temp_path)
        except FaceRecognitionError as e:
            return {
                "name": "No Face Detected",
                "confidence": 0.0,
                "status": "No face found in image"
            }
        
        # Get all embeddings from database
        embeddings = db.query(models.FaceEmbedding).all()
        
        if not embeddings:
            return {
                "name": "No Users",
                "confidence": 0.0,
                "status": "No users registered in system"
            }
        
        # Find best match
        best_match = None
        highest_score = 0.0
        
        for record in embeddings:
            stored_embedding = json.loads(record.embedding)
            score = cosine_similarity(new_embedding, stored_embedding)
            
            if score > highest_score:
                highest_score = score
                best_match = record.user_id
        
        # Check if match meets threshold
        if best_match and highest_score >= THRESHOLD:
            # Mark attendance
            status_result = mark_attendance(db, best_match)
            
            # Get user details
            user = db.query(models.User).filter(
                models.User.id == best_match
            ).first()
            
            logger.info(f"Face recognized: {user.name} with confidence {highest_score:.3f}")
            
            return {
                "name": user.name,
                "confidence": round(highest_score, 3),
                "status": status_result,
                "user_id": user.id
            }
        
        logger.info(f"No match found. Best confidence: {highest_score:.3f}")
        
        return {
            "name": "Unknown",
            "confidence": round(highest_score, 3),
            "status": "No Match",
            "threshold": THRESHOLD
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in recognition: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# Get all users
@app.get("/users/")
async def get_users(db: Session = Depends(get_db)):
    """
    Get all registered users
    """
    try:
        users = db.query(models.User).all()
        return [
            {
                "id": u.id, 
                "name": u.name, 
                "created_at": u.created_at.isoformat() if u.created_at else None
            } 
            for u in users
        ]
    except Exception as e:
        logger.error(f"Error fetching users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching users"
        )

# Get today's attendance
@app.get("/attendance/today")
async def get_today_attendance_endpoint(db: Session = Depends(get_db)):
    """
    Get all attendance records for today
    """
    try:
        attendance_records = get_today_attendance(db)
        
        result = []
        for att in attendance_records:
            user = db.query(models.User).filter(
                models.User.id == att.user_id
            ).first()
            
            result.append({
                "attendance_id": att.id,
                "user_id": att.user_id,
                "name": user.name if user else "Unknown",
                "date": att.date.isoformat(),
                "timestamp": att.timestamp.isoformat() if att.timestamp else None
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching attendance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching attendance"
        )

# Get user by ID
@app.get("/users/{user_id}")
async def get_user(user_id: int, db: Session = Depends(get_db)):
    """
    Get user details by ID
    """
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get user's embeddings
    embeddings = db.query(models.FaceEmbedding).filter(
        models.FaceEmbedding.user_id == user_id
    ).all()
    
    # Get user's attendance history
    attendance = db.query(models.Attendance).filter(
        models.Attendance.user_id == user_id
    ).order_by(models.Attendance.date.desc()).limit(10).all()
    
    return {
        "id": user.id,
        "name": user.name,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "embeddings_count": len(embeddings),
        "recent_attendance": [
            {
                "date": a.date.isoformat(),
                "timestamp": a.timestamp.isoformat() if a.timestamp else None
            }
            for a in attendance
        ]
    }

# Delete user
@app.delete("/users/{user_id}")
async def delete_user(user_id: int, db: Session = Depends(get_db)):
    """
    Delete user and all associated data
    """
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    try:
        # Delete user (cascade will handle embeddings and attendance)
        db.delete(user)
        db.commit()
        
        logger.info(f"User {user_id} deleted successfully")
        return {"message": "User deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting user: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting user"
        )