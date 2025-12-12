# # auth.py
# from passlib.context import CryptContext
# from datetime import datetime, timedelta
# import jwt
# from fastapi import HTTPException, Security, Depends
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from sqlalchemy.orm import Session
# from models import User
# from db import get_db

# SECRET_KEY = "MNCZBCASBHCBASHCBJSANCJKSNCJKN"
# ALGORITHM = "HS256"

# # Use Argon2 (bcrypt has 72-byte password limit)
# pwd_context = CryptContext(
#     schemes=["argon2"],
#     deprecated="auto"
# )

# # Bearer token security scheme
# security = HTTPBearer()

# def hash_password(password: str):
#     """Hash any length password safely."""
#     return pwd_context.hash(password)

# def verify_password(password: str, hashed_password: str):
#     """Verify password using Argon2."""
#     return pwd_context.verify(password, hashed_password)

# def create_access_token(data: dict, expires_minutes: int = 60):
#     """Create JWT token."""
#     to_encode = data.copy()
#     expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
#     to_encode.update({"exp": expire})
#     return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
#     """Verify Bearer token and return payload."""
#     try:
#         token = credentials.credentials
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         return payload
#     except jwt.ExpiredSignatureError:
#         raise HTTPException(status_code=401, detail="Token has expired")
#     except jwt.JWTError:
#         raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# def get_current_user(
#     payload: dict = Depends(verify_token),
#     db: Session = Depends(get_db)
# ) -> User:
#     """Get current authenticated user from token."""
#     user_id = payload.get("user_id")
#     if not user_id:
#         raise HTTPException(status_code=401, detail="Invalid token payload")
    
#     user = db.query(User).filter(User.id == user_id).first()
#     if not user:
#         raise HTTPException(status_code=401, detail="User not found")
    
#     return user

# def get_current_admin_user(
#     current_user: User = Depends(get_current_user)
# ) -> User:
#     """Get current authenticated admin user."""
#     if not current_user.is_admin:
#         raise HTTPException(status_code=403, detail="Not enough permissions. Admin access required.")
#     return current_user


# def decode_token(token: str) -> dict:
#     """Decode JWT token without verification (useful for testing)."""
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": False})
#         return payload
#     except jwt.JWTError:
#         raise HTTPException(status_code=401, detail="Invalid token")






# auth.py
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from models import User
from db import get_db

# -----------------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------------

SECRET_KEY = "MNCZBCASBHCBASHCBJSANCJKSNCJKN"
ALGORITHM = "HS256"

pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto"
)

security = HTTPBearer()

# -----------------------------------------------------------------------------------
# PASSWORD HELPERS
# -----------------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Hash password using Argon2."""
    return pwd_context.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password against Argon2 hash."""
    return pwd_context.verify(password, hashed_password)


# -----------------------------------------------------------------------------------
# JWT HELPERS
# -----------------------------------------------------------------------------------

def create_access_token(data: dict, expires_minutes: int = 60) -> str:
    """Create signed JWT token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    """Verify JWT token signature + expiry and return payload."""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired.")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication token.")


def decode_token(token: str) -> dict:
    """
    Decode JWT token WITHOUT verifying expiry.

    ⚠️ WARNING:
    - This is NOT secure for authentication.
    - Use only for debugging or admin analytics.
    """
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": False})
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token format.")


# -----------------------------------------------------------------------------------
# USER HELPERS
# -----------------------------------------------------------------------------------

def get_current_user(
    payload: dict = Depends(verify_token),
    db: Session = Depends(get_db)
) -> User:
    """Return user object from verified token."""
    user_id = payload.get("user_id")

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload.")

    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=401, detail="User not found.")

    return user


def get_current_admin_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Authenticate admin using decode_token() — no expiry check.
    Suitable when you don't want admins to be logged out frequently.
    """

    raw_token = credentials.credentials
    payload = decode_token(raw_token)

    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload.")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="Admin user not found.")

    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required.")

    return user







