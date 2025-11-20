
import uuid
import base64
import tempfile
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select
from pydantic import BaseModel
from pydub import AudioSegment
import speech_recognition as sr
from contextlib import asynccontextmanager
import os 
from dotenv import load_dotenv

from models import ChatMessage
from db import get_session, create_db_and_tables
from updated_rag import MasterChefAssistant
from elevenlabs_functions import speak_text_to_stream
load_dotenv()
ELEVENLABS_VOICE_ID=os.getenv("ELEVENLABS_VOICE_ID", "WV7clvf1VUCp942OSohW")

chef: MasterChefAssistant | None = None

# ---------------- FastAPI Setup ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global chef
    create_db_and_tables()
    chef = MasterChefAssistant()
    if not chef.initialize():
        print("WARN: Assistant failed to initialize")
        chef = None
    yield

app = FastAPI(title="Master Chef API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# ---------------- Schemas ----------------
class InitChatResponse(BaseModel):
    user_id: str
    text: str
    audio_base64: str

class AskQuestionRequest(BaseModel):
    text: str = ""
    voice_base64: str = ""

class AskQuestionResponse(BaseModel):
    chat: list
    text: str
    audio_base64: str

# ---------------- Helper Functions ----------------
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

# ---------------- Routes ----------------
@app.get("/init_chat", response_model=InitChatResponse)
def init_chat(session: Session = Depends(get_session)):
    ai_greet = "Hello! I'm your Master Chef voice assistant. I can guide you step-by-step through any recipe today."
    user_id = str(uuid.uuid4())

    audio_stream = speak_text_to_stream(ai_greet,ELEVENLABS_VOICE_ID)
    audio_bytes = audio_stream.getvalue()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    # Save initial bot message
    chat_row = ChatMessage(user_id=user_id)
    chat_row.append_chat("model", ai_greet)
    session.add(chat_row)
    session.commit()
    session.refresh(chat_row)

    return InitChatResponse(user_id=user_id, text=ai_greet, audio_base64=audio_base64)

@app.post("/ask_question_loop", response_model=AskQuestionResponse)
def ask_question_loop(request: AskQuestionRequest, user_id: str = Query(...), session: Session = Depends(get_session)):
    # Convert voice to text if text is empty
    question_text = request.text.strip()
    if question_text == "" and request.voice_base64:
        question_text = voice_to_text_mp3(request.voice_base64)
    elif question_text == "" and request.voice_base64 == "":
        raise HTTPException(status_code=400, detail="Please send audio or text!")
    
    if not question_text:
        raise HTTPException(status_code=400, detail="Could not understand input!")

    # Fetch existing chat row
    stmt = select(ChatMessage).where(ChatMessage.user_id == user_id)
    chat_row = session.exec(stmt).first()
    if not chat_row:
        chat_row = ChatMessage(user_id=user_id)

    # Append user message
    chat_row.append_chat("user", question_text)

    # Generate AI response
    response_text = chef.mentor_answer(user_id, question_text, session)
    response_text_loop = response_text

    # Append bot message
    chat_row.append_chat("model", response_text_loop)

    # Save to DB
    session.add(chat_row)
    session.commit()
    session.refresh(chat_row)

    # Convert AI response to Base64
    # audio_stream = speak_text_to_stream(response_text_loop, 'O483h7ZB7zKaA4JmK9Wv')
    audio_stream = speak_text_to_stream(response_text_loop, ELEVENLABS_VOICE_ID)
    audio_bytes = audio_stream.getvalue()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    return AskQuestionResponse(chat=chat_row.get_chat(), text=response_text_loop, audio_base64=audio_base64)

@app.get("/chat/messages", response_model=list)
def get_chat_messages(user_id: str, session: Session = Depends(get_session)):
    stmt = select(ChatMessage).where(ChatMessage.user_id == user_id)
    chat_row = session.exec(stmt).first()
    if not chat_row:
        return []
    return chat_row.get_chat()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)