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
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
    inspect as sqla_inspect,
    text,
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
    user_id: Mapped[str] = mapped_column(String(64), index=True, default="anonymous")
    name: Mapped[str] = mapped_column(String(255))
    parquet_path: Mapped[str] = mapped_column(String(512))
    columns: Mapped[list] = mapped_column(JSON)
    row_count: Mapped[int] = mapped_column(Integer)
    size_bytes: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now_utc)


class Chat(Base):
    __tablename__ = "chats"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, default="anonymous")
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(255), default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now_utc)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="chat",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
    dashboard_charts: Mapped[list["DashboardChart"]] = relationship(
        back_populates="chat",
        cascade="all, delete-orphan",
        order_by="DashboardChart.position",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    chat_id: Mapped[str] = mapped_column(String(36), ForeignKey("chats.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(16))  # user|assistant|tool
    content: Mapped[str] = mapped_column(Text, default="")
    pinned: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
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
    plotly_charts: Mapped[list["PlotlyChart"]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        order_by="PlotlyChart.created_at",
    )


class Chart(Base):
    __tablename__ = "charts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    message_id: Mapped[str] = mapped_column(String(36), ForeignKey("messages.id", ondelete="CASCADE"))
    # PNG bytes kept in the DB so the API doesn't need a persistent filesystem.
    # ~20 KB per chart — fine for an internal tool's volume.
    image_bytes: Mapped[bytes] = mapped_column(LargeBinary, default=b"")
    # Deprecated. Kept only to satisfy the legacy NOT NULL constraint on
    # existing SQLite DBs (SQLite can't drop columns easily). Always written as
    # empty string for new rows; reads come from image_bytes.
    path: Mapped[str] = mapped_column(String(512), default="")
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


class PlotlyChart(Base):
    __tablename__ = "plotly_charts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    message_id: Mapped[str] = mapped_column(String(36), ForeignKey("messages.id", ondelete="CASCADE"))
    spec_json: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(String(512), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now_utc)

    message: Mapped["Message"] = relationship(back_populates="plotly_charts")


class DashboardChart(Base):
    __tablename__ = "dashboard_charts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    chat_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("chats.id", ondelete="CASCADE"), index=True
    )
    spec_json: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(String(512), default="")
    position: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now_utc)

    chat: Mapped["Chat"] = relationship(back_populates="dashboard_charts")


class UserSettings(Base):
    """Per-user provider/model configuration (overrides env vars)."""
    __tablename__ = "user_settings"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    settings_json: Mapped[str] = mapped_column(Text, default="{}")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now_utc)


# ─── Engine + session factory ────────────────────────────────────────────────

_engine_kwargs: dict = {"future": True}
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db() -> None:  # noqa: C901
    """Create any missing tables and run any tiny in-place migrations.

    Idempotent and safe to call on startup. Real schema migrations should
    move to Alembic once the project has more than one deployed instance.
    """
    Base.metadata.create_all(engine)

    insp = sqla_inspect(engine)

    # Migration: dashboard_charts table (added post-MVP).
    if "dashboard_charts" not in insp.get_table_names():
        Base.metadata.tables["dashboard_charts"].create(engine)

    # Migration: messages.pinned (added after initial rollout).
    msg_cols = {c["name"] for c in insp.get_columns("messages")}
    if "pinned" not in msg_cols:
        with engine.begin() as conn:
            conn.execute(text(
                "ALTER TABLE messages ADD COLUMN pinned BOOLEAN NOT NULL DEFAULT 1"
            ))

    # Migration: user_id on datasets + chats. Existing rows get "anonymous".
    for table_name in ("datasets", "chats"):
        cols = {c["name"] for c in insp.get_columns(table_name)}
        if "user_id" not in cols:
            with engine.begin() as conn:
                conn.execute(text(
                    f"ALTER TABLE {table_name} ADD COLUMN user_id VARCHAR(64) "
                    f"NOT NULL DEFAULT 'anonymous'"
                ))

    # Migration: charts.image_bytes — when the column was added we may have
    # legacy rows with `path` pointing at on-disk PNGs. Backfill those once.
    chart_cols = {c["name"] for c in insp.get_columns("charts")}
    legacy_path_col_exists = "path" in chart_cols
    if "image_bytes" not in chart_cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE charts ADD COLUMN image_bytes BLOB"))
        if legacy_path_col_exists:
            from pathlib import Path as _P
            with engine.begin() as conn:
                rows = conn.execute(text(
                    "SELECT id, path FROM charts WHERE image_bytes IS NULL OR length(image_bytes) = 0"
                )).fetchall()
                for cid, p in rows:
                    if p and _P(p).exists():
                        try:
                            data = _P(p).read_bytes()
                        except OSError:
                            continue
                        conn.execute(
                            text("UPDATE charts SET image_bytes = :b WHERE id = :id"),
                            {"b": data, "id": cid},
                        )


def get_session() -> Session:
    """Caller is responsible for closing (use with-statement or try/finally)."""
    return SessionLocal()
