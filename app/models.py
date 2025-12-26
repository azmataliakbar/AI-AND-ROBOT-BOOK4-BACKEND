from datetime import datetime
from typing import Optional, List
from sqlalchemy import String, Integer, Float, DateTime, Text, ForeignKey, LargeBinary
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base


class Chapter(Base):
    __tablename__ = "chapters"

    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String, index=True)
    module: Mapped[int] = mapped_column(Integer)  # 1-4
    chapter_number: Mapped[int] = mapped_column(Integer)  # 1-37 for the entire book
    content: Mapped[str] = mapped_column(Text)
    word_count: Mapped[int] = mapped_column(Integer)
    estimated_reading_time: Mapped[int] = mapped_column(Integer)  # in minutes
    embedding_vector: Mapped[Optional[bytes]] = mapped_column(LargeBinary)  # Vector representation for semantic search
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    chat_histories: Mapped[List["ChatHistory"]] = relationship("ChatHistory", back_populates="chapter")
    progress_trackers: Mapped[List["ProgressTracker"]] = relationship("ProgressTracker", back_populates="chapter")


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    name: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    chat_histories: Mapped[List["ChatHistory"]] = relationship("ChatHistory", back_populates="user")
    progress_trackers: Mapped[List["ProgressTracker"]] = relationship("ProgressTracker", back_populates="user")


class ChatHistory(Base):
    __tablename__ = "chat_histories"

    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"))
    chapter_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("chapters.id"), nullable=True)
    user_query: Mapped[str] = mapped_column(Text)
    ai_response: Mapped[str] = mapped_column(Text)
    confidence_score: Mapped[float] = mapped_column(Float)  # 0.0-1.0
    source_type: Mapped[str] = mapped_column(String)  # vector, fallback, mixed
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    feedback_rating: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 1-5

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="chat_histories")
    chapter: Mapped[Optional["Chapter"]] = relationship("Chapter", back_populates="chat_histories")


class ProgressTracker(Base):
    __tablename__ = "progress_trackers"

    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"))
    chapter_id: Mapped[str] = mapped_column(String, ForeignKey("chapters.id"))
    completion_percentage: Mapped[float] = mapped_column(Float)  # 0.0-100.0
    last_read_position: Mapped[int] = mapped_column(Integer, default=0)  # Last scroll position or paragraph read
    time_spent: Mapped[int] = mapped_column(Integer, default=0)  # Time spent reading in seconds
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)  # When the chapter was completed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="progress_trackers")
    chapter: Mapped["Chapter"] = relationship("Chapter", back_populates="progress_trackers")


class EmbeddingCache(Base):
    __tablename__ = "embedding_caches"

    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    chapter_id: Mapped[str] = mapped_column(String, ForeignKey("chapters.id"))
    section_id: Mapped[str] = mapped_column(String)  # Identifier for the specific section within the chapter
    content_snippet: Mapped[str] = mapped_column(Text)  # The text content that was embedded
    embedding_vector: Mapped[bytes] = mapped_column(LargeBinary)  # Vector representation of the content
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    chapter: Mapped["Chapter"] = relationship("Chapter")