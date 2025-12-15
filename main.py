
# import uuid
# import base64
# import tempfile
# from datetime import datetime
# from pathlib import Path
# from fastapi import Depends, FastAPI, HTTPException, Query, UploadFile, File
# from pydantic import BaseModel
# from pydub import AudioSegment
# import speech_recognition as sr
# from fastapi.middleware.cors import CORSMiddleware
# from updated_rag import MasterChefAssistant
# from elevenlabs_functions import speak_text_to_stream
# from contextlib import asynccontextmanager

# from sqlmodel import Session, select
# import os
# from auth import hash_password, verify_password, create_access_token, get_current_admin_user
# from schemas import RegisterUser, LoginUser, UserOut
# from models import User, UploadedPDF, PasswordResetToken
# from db import get_db
# from schemas import ForgotPasswordRequest, ResetPasswordRequest


# from dotenv import load_dotenv
# load_dotenv()
# ELEVENLABS_VOICE_ID=os.getenv("ELEVENLABS_VOICE_ID", "O483h7ZB7zKaA4JmK9Wv")



# from db import create_db_and_tables  # import DB helpers
# from models import ChatMessage  # import models
# chef: MasterChefAssistant | None = None


# import secrets
# import hashlib
# from datetime import datetime, timedelta

# RESET_TOKEN_EXP_MINUTES = 15

# def generate_reset_token() -> str:
#     return secrets.token_urlsafe(32)

# def hash_reset_token(token: str) -> str:
#     return hashlib.sha256(token.encode()).hexdigest()



# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # startup
#     try:
#         create_db_and_tables()
#     except Exception as e:
#         print(f"WARN: DB init failed: {e}")
#     global chef
#     chef = MasterChefAssistant()
#     ok = chef.initialize()
#     if not ok:
#         print("WARN: Assistant failed to initialize")
#         chef = None
#     yield

# app = FastAPI(title="Master Chef Continuous Voice API", lifespan=lifespan, docs_url="/docs", redoc_url="/redoc")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Add your frontend URLs
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods including GET, POST, PUT, DELETE, etc.
#     allow_headers=["*"],  # Allows all headers
#     expose_headers=["*"]
# )

# class InitChatResponse(BaseModel):
#     user_id: str
#     text: str
#     audio_base64: str

# class AskQuestionRequest(BaseModel):
#     user_id: str
#     text: str
#     voice_base64: str    # MP3 Base64

# class AskQuestionResponse(BaseModel):
#     text: str
#     audio_base64: str

# # # Initialize Assistant
# # chef = MasterChefAssistant()
# # if not chef.initialize():
# #     raise RuntimeError("Failed to initialize MasterChefAssistant")

# # -----------------------------
# # Helper: Convert MP3 Base64 → WAV temp file
# # -----------------------------
# def mp3_base64_to_wav_file(mp3_base64: str) -> str:
#     audio_bytes = base64.b64decode(mp3_base64)
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
#         tmp_mp3.write(audio_bytes)
#         mp3_path = tmp_mp3.name

#     wav_path = mp3_path.replace(".mp3", ".wav")
#     audio = AudioSegment.from_mp3(mp3_path)
#     audio.export(wav_path, format="wav")
#     return wav_path

# # -----------------------------
# # Voice → Text
# # -----------------------------
# def voice_to_text_mp3(voice_base64: str) -> str:
#     wav_path = mp3_base64_to_wav_file(voice_base64)
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(wav_path) as source:
#         audio_data = recognizer.record(source)
#         try:
#             return recognizer.recognize_google(audio_data, language="en-US")
#         except sr.UnknownValueError:
#             return ""
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Speech recognition error: {e}")

# # -----------------------------
# # Init Chat API
# # -----------------------------
# @app.get("/init_chat")
# def initialize_chat(session: Session = Depends(get_db),):
#     ai_greet = "Hello! I'm your AI Master Chef Rosendale. I can guide you step-by-step through any recipe or cooking technique today."

#     # Convert Initial response to Base64 MP3
#     audio_stream = speak_text_to_stream(ai_greet, ELEVENLABS_VOICE_ID)
#     audio_bytes = audio_stream.getvalue()
#     audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

#     # Generating Unique Id
#     user_id = str(uuid.uuid4()) + str(int(datetime.timestamp(datetime.now())))

#     # Saving to DB
#     msg = ChatMessage(user_id=user_id, type='model', text=ai_greet)
#     session.add(msg)
#     session.commit()
#     session.refresh(msg)

#     return InitChatResponse(user_id=user_id, text=ai_greet, audio_base64=audio_base64)


# @app.get("/chat/messages")
# def list_messages(
#     session: Session = Depends(get_db),
#     user_id: str | None = None,
#     limit: int = Query(100, le=100),
# ):
#     stmt = select(ChatMessage).order_by(ChatMessage.created_at.desc()).limit(limit)
#     if user_id is not None:
#         stmt = stmt.where(ChatMessage.user_id == user_id)
#     return session.exec(stmt).all()


# # -----------------------------
# # Ask Question API
# # -----------------------------
# @app.post("/ask_question_loop", response_model=AskQuestionResponse)
# def ask_question_loop(request: AskQuestionRequest, session: Session = Depends(get_db),):
#     user_text = request.text.strip()
#     user_base64 = request.voice_base64
#     question_text = ''
#     if user_text == '' and user_base64 != '':
#         question_text = voice_to_text_mp3(user_base64)
#         print('==question_text===',question_text)
   
#     elif user_text != '' and user_base64 == '':
#         question_text = user_text
   
#     elif user_text == '' and user_base64 == '':
#         raise HTTPException(status_code=400, detail="Please send audio or text!")
 
#     if not question_text:
#         raise HTTPException(status_code=400, detail="Could not understand the audio!")
 
#     user_msg = ChatMessage(user_id=request.user_id, type='user', text=question_text)
#     session.add(user_msg)
#     session.commit()
#     session.refresh(user_msg)
 
#     # Generate AI response
#     chef.set_dish(question_text)
#     response_text = chef.mentor_answer(request.user_id, question_text, session)
 
#     # Add "Have any other questions?" to response
#     response_text_loop = response_text
 
#     # Convert AI response to Base64 MP3
#     audio_stream = speak_text_to_stream(response_text_loop, ELEVENLABS_VOICE_ID)
#     audio_bytes = audio_stream.getvalue()
#     audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
 
#     agent_msg = ChatMessage(user_id=request.user_id, type='model', text=response_text_loop)
#     session.add(agent_msg)
#     session.commit()
#     session.refresh(agent_msg)
 
#     return AskQuestionResponse(text=response_text_loop, audio_base64=audio_base64)


# @app.post("/register", response_model=UserOut)
# def register_user(data: RegisterUser, db: Session = Depends(get_db)):

#     # Basic validation
#     if not data.username or not data.email or not data.password:
#         raise HTTPException(
#             status_code=400,
#             detail="Username, email, and password are required"
#         )

#     # Check if email already exists
#     user_exists = db.query(User).filter(User.email == data.email).first()
#     if user_exists:
#         raise HTTPException(
#             status_code=400,
#             detail="Email already exists"
#         )

#     # Check if username already exists
#     username_exists = db.query(User).filter(User.username == data.username).first()
#     if username_exists:
#         raise HTTPException(
#             status_code=400,
#             detail="Username already exists"
#         )

#     # Create new user
#     new_user = User(
#         username=data.username,
#         email=data.email,
#         password=hash_password(data.password),
#         is_admin=True
#     )

#     db.add(new_user)
#     db.commit()
#     db.refresh(new_user)

#     return new_user



# @app.post("/login")
# def login_user(data: LoginUser, db: Session = Depends(get_db)):
#     user = db.query(User).filter(User.email == data.email).first()
 
#     if not user:
#         return {
#             "status": False,
#             "message": "Invalid email or password",
#             "data": None
#         }
 
#     if not user.is_admin:
#         return {
#             "status": False,
#             "message": "Only admin users are allowed to login",
#             "data": None
#         }
 
#     if not verify_password(data.password, user.password):
#         return {
#             "status": False,
#             "message": "Invalid password",
#             "data": None
#         }
 
#     # Update last login
#     user.last_login = datetime.utcnow()
#     db.commit()
 
#     token = create_access_token({"user_id": user.id, "email": user.email})
 
#     return {
#         "status": True,
#         "message": "Login successful",
#         "data": {
#             "token": token,
#             "last_login": user.last_login,
#             "user": {
#                 "id": user.id,
#                 "username": user.username,
#                 "email": user.email,
#                 "is_admin": user.is_admin
#             }
#         }
#     }
    
    
    
# from fastapi import BackgroundTasks

# @app.post("/forgot-password")
# def forgot_password(
#     data: ForgotPasswordRequest,
#     background_tasks: BackgroundTasks,
#     db: Session = Depends(get_db)
# ):
#     user = db.query(User).filter(User.email == data.email).first()

#     # Always return same message (avoid email enumeration)
#     response = {
#         "status": True,
#         "message": "If the email exists, a password reset link has been sent."
#     }

#     if not user:
#         return response

#     # Invalidate previous tokens
#     db.query(PasswordResetToken).filter(
#         PasswordResetToken.user_id == user.id,
#         PasswordResetToken.is_used == False
#     ).update({"is_used": True})

#     # Create token
#     plain_token = generate_reset_token()
#     token_hash = hash_reset_token(plain_token)

#     reset_token = PasswordResetToken(
#         user_id=user.id,
#         token_hash=token_hash,
#         expires_at=datetime.utcnow() + timedelta(minutes=RESET_TOKEN_EXP_MINUTES)
#     )

#     db.add(reset_token)
#     db.commit()

#     # Send email (replace with real email service)
#     reset_link = f"https://yourfrontend.com/reset-password?token={plain_token}"
#     background_tasks.add_task(
#         print, f"Password reset link: {reset_link}"
#     )

#     return response



# @app.post("/reset-password")
# def reset_password(
#     data: ResetPasswordRequest,
#     db: Session = Depends(get_db)
# ):
#     token_hash = hash_reset_token(data.token)

#     reset_token = db.query(PasswordResetToken).filter(
#         PasswordResetToken.token_hash == token_hash,
#         PasswordResetToken.is_used == False,
#         PasswordResetToken.expires_at > datetime.utcnow()
#     ).first()

#     if not reset_token:
#         raise HTTPException(status_code=400, detail="Invalid or expired token")

#     user = db.query(User).filter(User.id == reset_token.user_id).first()
#     if not user:
#         raise HTTPException(status_code=400, detail="User not found")

#     # Update password
#     user.password = hash_password(data.new_password)

#     # Mark token as used
#     reset_token.is_used = True

#     db.commit()

#     return {
#         "status": True,
#         "message": "Password reset successful"
#     }



# class PDFUploadResponse(BaseModel):
#     status: bool
#     message: str
#     data: dict = None


# @app.post("/rag/update", response_model=PDFUploadResponse)
# async def update_rag_with_pdf(
#     file: UploadFile = File(...),
#     current_user: User = Depends(get_current_admin_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     Admin endpoint to upload a new PDF and add it to the RAG system.
#     Checks for duplicate PDFs based on filename.
#     """
#     # Validate file type
#     if not file.filename.lower().endswith('.pdf'):
#         raise HTTPException(
#             status_code=400,
#             detail="Invalid file type. Only PDF files are allowed."
#         )
    
#     # Check if PDF already exists in database
#     existing_pdf = db.query(UploadedPDF).filter(UploadedPDF.filename == file.filename).first()
#     if existing_pdf:
#         raise HTTPException(
#             status_code=400,
#             detail=f"PDF with filename '{file.filename}' already exists in the database."
#         )
    
#     # Ensure Recipe directory exists
#     recipe_dir = Path("Recipe")
#     recipe_dir.mkdir(exist_ok=True)
    
#     # Save uploaded file
#     file_path = recipe_dir / file.filename
    
#     # Check if file already exists on disk
#     if file_path.exists():
#         raise HTTPException(
#             status_code=400,
#             detail=f"PDF file '{file.filename}' already exists on the server."
#         )
    
#     try:
#         # Save file to disk
#         content = await file.read()
#         with open(file_path, "wb") as f:
#             f.write(content)
        
#         # Ensure chef instance is initialized
#         global chef
#         if chef is None:
#             chef = MasterChefAssistant()
#             chef.initialize()
        
#         # Add PDF to RAG system
#         result = chef.add_pdf_to_rag(file_path)
        
#         if not result.get("success", False):
#             # Clean up file if RAG processing failed
#             if file_path.exists():
#                 file_path.unlink()
#             raise HTTPException(
#                 status_code=500,
#                 detail=result.get("message", "Failed to process PDF in RAG system")
#             )
        
#         # Save PDF record to database
#         uploaded_pdf = UploadedPDF(
#             filename=file.filename,
#             file_path=str(file_path),
#             uploaded_by=current_user.id
#         )
#         db.add(uploaded_pdf)
#         db.commit()
#         db.refresh(uploaded_pdf)
        
#         return PDFUploadResponse(
#             status=True,
#             message="PDF successfully uploaded and added to RAG system",
#             data={
#                 "filename": file.filename,
#                 "file_path": str(file_path),
#                 "pages": result.get("pages", 0),
#                 "chunks": result.get("chunks", 0),
#                 "uploaded_at": uploaded_pdf.uploaded_at.isoformat() if uploaded_pdf.uploaded_at else None,
#                 "uploaded_by": current_user.username
#             }
#         )
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         # Clean up file if something went wrong
#         if file_path.exists():
#             file_path.unlink()
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error processing PDF: {str(e)}"
#         )


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)





































import uuid
import base64
import tempfile
from datetime import datetime
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException, Query, UploadFile, File
from pydantic import BaseModel
from pydub import AudioSegment
import speech_recognition as sr
from fastapi.middleware.cors import CORSMiddleware
from updated_rag import MasterChefAssistant
from elevenlabs_functions import speak_text_to_stream
from contextlib import asynccontextmanager

from sqlmodel import Session, select
import os
from auth import hash_password, verify_password, create_access_token, get_current_admin_user
from schemas import RegisterUser, LoginUser, UserOut
from models import User, UploadedPDF, PasswordResetToken
from db import get_db
from schemas import ForgotPasswordRequest, ResetPasswordRequest

# OTP Email imports
import smtplib
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv
load_dotenv()
ELEVENLABS_VOICE_ID=os.getenv("ELEVENLABS_VOICE_ID", "O483h7ZB7zKaA4JmK9Wv")

# Email Configuration for OTP
SENDER_EMAIL = "vishalsharma07ms@gmail.com"
SENDER_PASSWORD = ("sbvp cmvl hrxf qurc")  # Add to .env file
OTP_EXP_MINUTES = 10

from db import create_db_and_tables
from models import ChatMessage
chef: MasterChefAssistant | None = None

import secrets
import hashlib
from datetime import timedelta

def generate_reset_token() -> str:
    return secrets.token_urlsafe(32)

def hash_reset_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()

# OTP Helper Functions
def generate_otp():
    """Generate 6-digit OTP"""
    return str(random.randint(100000, 999999))

def send_otp_email(recipient_email: str, otp: str):
    """Send OTP via email"""
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = recipient_email
    msg['Subject'] = "Password Reset OTP - Master Chef"
    
    body = f"""
Hello,

You requested to reset your password. Use the OTP below to proceed:

OTP: {otp}

This OTP is valid for {OTP_EXP_MINUTES} minutes.
Do not share this code with anyone.

If you didn't request this, please ignore this email.

Best regards,
Master Chef Team
    """
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        
        server.send_message(msg)
        server.quit()
        print(f"OTP sent to {recipient_email}")
        return True
    except Exception as e:
        raise
        print(f"Email error: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    try:
        create_db_and_tables()
    except Exception as e:
        print(f"WARN: DB init failed: {e}")
    global chef
    chef = MasterChefAssistant()
    ok = chef.initialize()
    if not ok:
        print("WARN: Assistant failed to initialize")
        chef = None
    yield

app = FastAPI(title="Master Chef Continuous Voice API", lifespan=lifespan, docs_url="/docs", redoc_url="/redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

class InitChatResponse(BaseModel):
    user_id: str
    text: str
    audio_base64: str

class AskQuestionRequest(BaseModel):
    user_id: str
    text: str
    voice_base64: str

class AskQuestionResponse(BaseModel):
    text: str
    audio_base64: str

def mp3_base64_to_wav_file(mp3_base64: str) -> str:
    audio_bytes = base64.b64decode(mp3_base64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
        tmp_mp3.write(audio_bytes)
        mp3_path = tmp_mp3.name

    wav_path = mp3_path.replace(".mp3", ".wav")
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")
    return wav_path

def voice_to_text_mp3(voice_base64: str) -> str:
    wav_path = mp3_base64_to_wav_file(voice_base64)
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data, language="en-US")
        except sr.UnknownValueError:
            return ""
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Speech recognition error: {e}")

@app.get("/init_chat")
def initialize_chat(session: Session = Depends(get_db),):
    ai_greet = "Hello! I'm your AI Master Chef Rosendale. I can guide you step-by-step through any recipe or cooking technique today."

    audio_stream = speak_text_to_stream(ai_greet, ELEVENLABS_VOICE_ID)
    audio_bytes = audio_stream.getvalue()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    user_id = str(uuid.uuid4()) + str(int(datetime.timestamp(datetime.now())))

    msg = ChatMessage(user_id=user_id, type='model', text=ai_greet)
    session.add(msg)
    session.commit()
    session.refresh(msg)

    return InitChatResponse(user_id=user_id, text=ai_greet, audio_base64=audio_base64)

@app.get("/chat/messages")
def list_messages(
    session: Session = Depends(get_db),
    user_id: str | None = None,
    limit: int = Query(100, le=100),
):
    stmt = select(ChatMessage).order_by(ChatMessage.created_at.desc()).limit(limit)
    if user_id is not None:
        stmt = stmt.where(ChatMessage.user_id == user_id)
    return session.exec(stmt).all()

@app.post("/ask_question_loop", response_model=AskQuestionResponse)
def ask_question_loop(request: AskQuestionRequest, session: Session = Depends(get_db),):
    user_text = request.text.strip()
    user_base64 = request.voice_base64
    question_text = ''
    if user_text == '' and user_base64 != '':
        question_text = voice_to_text_mp3(user_base64)
        print('==question_text===',question_text)
   
    elif user_text != '' and user_base64 == '':
        question_text = user_text
   
    elif user_text == '' and user_base64 == '':
        raise HTTPException(status_code=400, detail="Please send audio or text!")
 
    if not question_text:
        raise HTTPException(status_code=400, detail="Could not understand the audio!")
 
    user_msg = ChatMessage(user_id=request.user_id, type='user', text=question_text)
    session.add(user_msg)
    session.commit()
    session.refresh(user_msg)
 
    chef.set_dish(question_text)
    response_text = chef.mentor_answer(request.user_id, question_text, session)
 
    response_text_loop = response_text
 
    audio_stream = speak_text_to_stream(response_text_loop, ELEVENLABS_VOICE_ID)
    audio_bytes = audio_stream.getvalue()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
 
    agent_msg = ChatMessage(user_id=request.user_id, type='model', text=response_text_loop)
    session.add(agent_msg)
    session.commit()
    session.refresh(agent_msg)
 
    return AskQuestionResponse(text=response_text_loop, audio_base64=audio_base64)

@app.post("/register", response_model=UserOut)
def register_user(data: RegisterUser, db: Session = Depends(get_db)):
    if not data.username or not data.email or not data.password:
        raise HTTPException(
            status_code=400,
            detail="Username, email, and password are required"
        )

    user_exists = db.query(User).filter(User.email == data.email).first()
    if user_exists:
        raise HTTPException(
            status_code=400,
            detail="Email already exists"
        )

    username_exists = db.query(User).filter(User.username == data.username).first()
    if username_exists:
        raise HTTPException(
            status_code=400,
            detail="Username already exists"
        )

    new_user = User(
        username=data.username,
        email=data.email,
        password=hash_password(data.password),
        is_admin=True
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user

@app.post("/login")
def login_user(data: LoginUser, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
 
    if not user:
        return {
            "status": False,
            "message": "Invalid email or password",
            "data": None
        }
 
    if not user.is_admin:
        return {
            "status": False,
            "message": "Only admin users are allowed to login",
            "data": None
        }
 
    if not verify_password(data.password, user.password):
        return {
            "status": False,
            "message": "Invalid password",
            "data": None
        }
 
    user.last_login = datetime.utcnow()
    db.commit()
 
    token = create_access_token({"user_id": user.id, "email": user.email})
 
    return {
        "status": True,
        "message": "Login successful",
        "data": {
            "token": token,
            "last_login": user.last_login,
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "is_admin": user.is_admin
            }
        }
    }

from fastapi import BackgroundTasks

@app.post("/forgot-password")
def forgot_password(
    data: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Step 1: Send OTP to user's email"""
    user = db.query(User).filter(User.email == data.email).first()

    response = {
        "status": True,
        "message": "If the email exists, a password reset OTP has been sent."
    }

    if not user:
        return response

    # Invalidate all previous unused tokens
    db.query(PasswordResetToken).filter(
        PasswordResetToken.user_id == user.id,
        PasswordResetToken.is_used == False
    ).update({"is_used": True})
    db.commit()

    # Generate OTP
    otp = generate_otp()
    token_hash = hash_reset_token(otp)

    # Save OTP in database
    reset_token = PasswordResetToken(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=datetime.utcnow() + timedelta(minutes=OTP_EXP_MINUTES)
    )

    db.add(reset_token)
    db.commit()

    # Send OTP email in background
    background_tasks.add_task(send_otp_email, user.email, otp)  # ✅ Fixed

    return response

@app.post("/reset-password")
def reset_password(
    data: ResetPasswordRequest,
    db: Session = Depends(get_db)
):
    """Step 2: Verify OTP and reset password"""
    
    # Find user
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid email or OTP")

    # Hash the provided OTP
    otp_hash = hash_reset_token(data.otp)

    # Find valid token
    reset_token = db.query(PasswordResetToken).filter(
        PasswordResetToken.user_id == user.id,
        PasswordResetToken.token_hash == otp_hash,
        PasswordResetToken.is_used == False,
        PasswordResetToken.expires_at > datetime.utcnow()
    ).first()

    if not reset_token:
        raise HTTPException(
            status_code=400, 
            detail="Invalid or expired OTP"
        )

    # Update password
    user.password = hash_password(data.new_password)

    # Mark token as used
    reset_token.is_used = True

    db.commit()

    return {
        "status": True,
        "message": "Password reset successful. You can now login with your new password."
    }

class PDFUploadResponse(BaseModel):
    status: bool
    message: str
    data: dict = None

@app.post("/rag/update", response_model=PDFUploadResponse)
async def update_rag_with_pdf(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Admin endpoint to upload a new PDF and add it to the RAG system.
    Checks for duplicate PDFs based on filename.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF files are allowed."
        )
    
    existing_pdf = db.query(UploadedPDF).filter(UploadedPDF.filename == file.filename).first()
    if existing_pdf:
        raise HTTPException(
            status_code=400,
            detail=f"PDF with filename '{file.filename}' already exists in the database."
        )
    
    recipe_dir = Path("Recipe")
    recipe_dir.mkdir(exist_ok=True)
    
    file_path = recipe_dir / file.filename
    
    if file_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"PDF file '{file.filename}' already exists on the server."
        )
    
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        global chef
        if chef is None:
            chef = MasterChefAssistant()
            chef.initialize()
        
        result = chef.add_pdf_to_rag(file_path)
        
        if not result.get("success", False):
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Failed to process PDF in RAG system")
            )
        
        uploaded_pdf = UploadedPDF(
            filename=file.filename,
            file_path=str(file_path),
            uploaded_by=current_user.id
        )
        db.add(uploaded_pdf)
        db.commit()
        db.refresh(uploaded_pdf)
        
        return PDFUploadResponse(
            status=True,
            message="PDF successfully uploaded and added to RAG system",
            data={
                "filename": file.filename,
                "file_path": str(file_path),
                "pages": result.get("pages", 0),
                "chunks": result.get("chunks", 0),
                "uploaded_at": uploaded_pdf.uploaded_at.isoformat() if uploaded_pdf.uploaded_at else None,
                "uploaded_by": current_user.username
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)