# from sqlmodel import SQLModel, create_engine, Session

# sqlite_url = "sqlite:///./database.db"
# engine = create_engine(sqlite_url, echo=True, connect_args={"check_same_thread": False})

# def create_db_and_tables() -> None:
#     SQLModel.metadata.create_all(engine)

# def get_session():
#     with Session(engine) as session:
#         yield session

# db.py
from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.orm import declarative_base

# ---------------------------------------------------
# DATABASE URL
# ---------------------------------------------------
DATABASE_URL = "sqlite:///./database.db"

# ---------------------------------------------------
# ENGINE - Works for SQLModel + SQLAlchemy
# ---------------------------------------------------
engine = create_engine(
    DATABASE_URL,
    echo=True,
    connect_args={"check_same_thread": False}  # needed for SQLite
)

# ---------------------------------------------------
# Base class for SQLAlchemy ORM models (User model)
# ---------------------------------------------------
Base = declarative_base()

# ---------------------------------------------------
# Create all tables: SQLModel + SQLAlchemy
# ---------------------------------------------------
def create_db_and_tables():
    from models import ChatMessage, User, UploadedPDF  # Import here to avoid circular imports

    # Create SQLModel tables
    SQLModel.metadata.create_all(engine)

    # Create SQLAlchemy ORM tables
    Base.metadata.create_all(bind=engine)

# ---------------------------------------------------
# FastAPI Dependency: Get DB Session
# ---------------------------------------------------
def get_db():
    with Session(engine) as session:
        yield session

