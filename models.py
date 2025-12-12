# app/models.py
from __future__ import annotations
from typing import Optional
import enum
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field

from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from db import Base
 

class ChatRole(str, enum.Enum):
    user = "user"
    model = "model"

class ChatMessageBase(SQLModel):
    type: ChatRole = Field(description="Sender role")
    text: str = Field(description="Message content")
    user_id: str = Field(description="User id whose conversation is this")

class ChatMessage(ChatMessageBase, table=True):
    __tablename__ = "chat_messages"
    id: Optional[int] = Field(default=None, primary_key=True)  # auto-increment PK
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp"
    )

class User(Base):
    __tablename__ = "users"
 
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    is_admin = Column(Boolean, default=True)  # default admin = True
    last_login = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class UploadedPDF(Base):
    __tablename__ = "uploaded_pdfs"
 
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True, nullable=False)
    file_path = Column(String, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    uploaded_by = Column(Integer, nullable=False)  # User ID who uploaded