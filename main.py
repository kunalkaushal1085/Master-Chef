
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
from models import User, UploadedPDF
from db import get_db


from dotenv import load_dotenv
load_dotenv()
ELEVENLABS_VOICE_ID=os.getenv("ELEVENLABS_VOICE_ID", "O483h7ZB7zKaA4JmK9Wv")



from db import create_db_and_tables  # import DB helpers
from models import ChatMessage  # import models
chef: MasterChefAssistant | None = None


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
    allow_origins=["*"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods including GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"]
)

class InitChatResponse(BaseModel):
    user_id: str
    text: str
    audio_base64: str

class AskQuestionRequest(BaseModel):
    user_id: str
    text: str
    voice_base64: str    # MP3 Base64

class AskQuestionResponse(BaseModel):
    text: str
    audio_base64: str

# # Initialize Assistant
# chef = MasterChefAssistant()
# if not chef.initialize():
#     raise RuntimeError("Failed to initialize MasterChefAssistant")

# -----------------------------
# Helper: Convert MP3 Base64 → WAV temp file
# -----------------------------
def mp3_base64_to_wav_file(mp3_base64: str) -> str:
    audio_bytes = base64.b64decode(mp3_base64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
        tmp_mp3.write(audio_bytes)
        mp3_path = tmp_mp3.name

    wav_path = mp3_path.replace(".mp3", ".wav")
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")
    return wav_path

# -----------------------------
# Voice → Text
# -----------------------------
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

# -----------------------------
# Init Chat API
# -----------------------------
@app.get("/init_chat")
def initialize_chat(session: Session = Depends(get_db),):
    ai_greet = "Hello! I'm your AI Master Chef Rosendale. I can guide you step-by-step through any recipe or cooking technique today."

    # Convert Initial response to Base64 MP3
    audio_stream = speak_text_to_stream(ai_greet, ELEVENLABS_VOICE_ID)
    audio_bytes = audio_stream.getvalue()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    # Generating Unique Id
    user_id = str(uuid.uuid4()) + str(int(datetime.timestamp(datetime.now())))

    # Saving to DB
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


# -----------------------------
# Ask Question API
# -----------------------------
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
 
    # Generate AI response
    chef.set_dish(question_text)
    response_text = chef.mentor_answer(request.user_id, question_text, session)
 
    # Add "Have any other questions?" to response
    response_text_loop = response_text
 
    # Convert AI response to Base64 MP3
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

    # Basic validation
    if not data.username or not data.email or not data.password:
        raise HTTPException(
            status_code=400,
            detail="Username, email, and password are required"
        )

    # Check if email already exists
    user_exists = db.query(User).filter(User.email == data.email).first()
    if user_exists:
        raise HTTPException(
            status_code=400,
            detail="Email already exists"
        )

    # Check if username already exists
    username_exists = db.query(User).filter(User.username == data.username).first()
    if username_exists:
        raise HTTPException(
            status_code=400,
            detail="Username already exists"
        )

    # Create new user
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
 
    # Update last login
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
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF files are allowed."
        )
    
    # Check if PDF already exists in database
    existing_pdf = db.query(UploadedPDF).filter(UploadedPDF.filename == file.filename).first()
    if existing_pdf:
        raise HTTPException(
            status_code=400,
            detail=f"PDF with filename '{file.filename}' already exists in the database."
        )
    
    # Ensure Recipe directory exists
    recipe_dir = Path("Recipe")
    recipe_dir.mkdir(exist_ok=True)
    
    # Save uploaded file
    file_path = recipe_dir / file.filename
    
    # Check if file already exists on disk
    if file_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"PDF file '{file.filename}' already exists on the server."
        )
    
    try:
        # Save file to disk
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Ensure chef instance is initialized
        global chef
        if chef is None:
            chef = MasterChefAssistant()
            chef.initialize()
        
        # Add PDF to RAG system
        result = chef.add_pdf_to_rag(file_path)
        
        if not result.get("success", False):
            # Clean up file if RAG processing failed
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Failed to process PDF in RAG system")
            )
        
        # Save PDF record to database
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
        # Clean up file if something went wrong
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
