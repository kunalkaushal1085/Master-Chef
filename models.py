from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime, timezone
import uuid
import json

class ChatMessage(SQLModel, table=True):
    __tablename__ = "chat_messages"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="User id")
    chat: str = Field(default="[]", description="Conversation stored as JSON string")
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp")

    def get_chat(self):
        return json.loads(self.chat)
    def append_chat(self, role: str, text: str):
        role_map = {"user": "user", "model": "bot"}
        chat = self.get_chat()
        chat.append({role_map[role]: text})
        self.chat = json.dumps(chat)
