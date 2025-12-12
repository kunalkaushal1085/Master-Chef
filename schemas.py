# schemas.py
from pydantic import BaseModel, EmailStr
 
class RegisterUser(BaseModel):
    username: str
    email: EmailStr
    password: str
 
class LoginUser(BaseModel):
    email: EmailStr
    password: str
 
from datetime import datetime

class UserOut(BaseModel):
    id: int
    username: str
    email: str
    is_admin: bool
    last_login: datetime | None = None

    class Config:
        from_attributes = True