# app/models.py
from __future__ import annotations
from typing import Optional
import enum
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field

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
