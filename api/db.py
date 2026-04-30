"""SQLAlchemy 2.x models + engine factory.

Default backend is SQLite (file at api/data.db). Set DATABASE_URL to switch
to Postgres (psycopg2 driver, Neon-compatible). Tables are auto-created on
first run via metadata.create_all() — no Alembic needed for the MVP.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

API_DIR = Path(__file__).parent
DEFAULT_DB_URL = f"sqlite:///{API_DIR / 'data.db'}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

# Files live on disk; the DB only holds paths.
DATA_DIR = Path(os.getenv("DATA_DIR", API_DIR / "data"))
DATASETS_DIR = DATA_DIR / "datasets"
CHARTS_DIR = DATA_DIR / "charts"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    parquet_path: Mapped[str] = mapped_column(String(512))
    columns: Mapped[list] = mapped_column(JSON)
    row_count: Mapped[int] = mapped_column(Integer)
    size_bytes: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now_utc)


class Chat(Base):
    __tablename__ = "chats"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(255), default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now_utc)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="chat",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    chat_id: Mapped[str] = mapped_column(String(36), ForeignKey("chats.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(16))  # user|assistant|tool
    content: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now_utc)

    chat: Mapped["Chat"] = relationship(back_populates="messages")
    charts: Mapped[list["Chart"]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        order_by="Chart.created_at",
    )
    snippets: Mapped[list["Snippet"]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        order_by="Snippet.created_at",
    )


class Chart(Base):
    __tablename__ = "charts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    message_id: Mapped[str] = mapped_column(String(36), ForeignKey("messages.id", ondelete="CASCADE"))
    path: Mapped[str] = mapped_column(String(512))
    title: Mapped[str] = mapped_column(String(512), default="")
    chart_type: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now_utc)

    message: Mapped["Message"] = relationship(back_populates="charts")


class Snippet(Base):
    __tablename__ = "snippets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    message_id: Mapped[str] = mapped_column(String(36), ForeignKey("messages.id", ondelete="CASCADE"))
    type: Mapped[str] = mapped_column(String(16))  # analysis|sql|chart
    code: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now_utc)

    message: Mapped["Message"] = relationship(back_populates="snippets")


# ─── Engine + session factory ────────────────────────────────────────────────

_engine_kwargs: dict = {"future": True}
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create any missing tables. Idempotent and safe to call on startup."""
    Base.metadata.create_all(engine)


def get_session() -> Session:
    """Caller is responsible for closing (use with-statement or try/finally)."""
    return SessionLocal()
