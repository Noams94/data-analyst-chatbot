"""SQLAlchemy 2.x models + engine factory.

Default backend is SQLite (file at api/data.db). Set DATABASE_URL to switch
to Postgres (psycopg2 driver, Neon-compatible). Tables are auto-created on
first run via metadata.create_all() — no Alembic needed for the MVP.
"""

from __future__ import annotations

import os
import sys
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

# Default DB lives outside ~/Documents to avoid iCloud-sync corruption of
# long-running SQLite connections (macOS evicts/syncs files under us, which
# leaves open file descriptors in a permanent "disk I/O error" state).
def _default_db_path() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "data_analyst_chatbot"
    else:
        base = API_DIR
    base.mkdir(parents=True, exist_ok=True)
    target = base / "data.db"
    # One-time migration from the old in-repo location.
    legacy = API_DIR / "data.db"
    if legacy.exists() and not target.exists():
        import shutil
        shutil.copy2(legacy, target)
    return target


DEFAULT_DB_URL = f"sqlite:///{_default_db_path()}"
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
    # 12-col grid layout. width: 1..12 cols, height: 1..6 row units (~120px each).
    layout_w: Mapped[int] = mapped_column(Integer, default=6)
    layout_h: Mapped[int] = mapped_column(Integer, default=2)
    chart_kind: Mapped[str] = mapped_column(String(32), default="plotly")  # plotly|kpi
    kpi_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Set when the chart was promoted from a finding — lets us remove it again
    # when the finding's "in dashboard" flag is turned off.
    finding_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("findings.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now_utc)

    chat: Mapped["Chat"] = relationship(back_populates="dashboard_charts")


class Finding(Base):
    """Structured insight recorded by the LLM. Replaces ad-hoc message-pinning
    as the unit of curation: a finding has a claim, evidence, optional chart,
    and explicit flags for inclusion in dashboard / report."""

    __tablename__ = "findings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    chat_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("chats.id", ondelete="CASCADE"), index=True
    )
    message_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )
    chart_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("plotly_charts.id", ondelete="SET NULL"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(512), default="")
    claim: Mapped[str] = mapped_column(Text, default="")
    evidence_json: Mapped[str] = mapped_column(Text, default="{}")
    confidence: Mapped[str] = mapped_column(String(16), default="medium")  # low|medium|high
    dimension: Mapped[str] = mapped_column(String(64), default="")  # e.g. "revenue", "retention"
    position: Mapped[int] = mapped_column(Integer, default=0)
    included_in_report: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    included_in_dashboard: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now_utc)


class ReportDraft(Base):
    """Auto-generated narrative draft for a chat's report. One row per chat."""

    __tablename__ = "report_drafts"

    chat_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("chats.id", ondelete="CASCADE"), primary_key=True
    )
    sections_json: Mapped[str] = mapped_column(Text, default="{}")
    style: Mapped[str] = mapped_column(String(32), default="executive")
    language: Mapped[str] = mapped_column(String(8), default="en")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now_utc)


class DashboardShare(Base):
    """Public read-only token for a chat's dashboard."""

    __tablename__ = "dashboard_shares"

    token: Mapped[str] = mapped_column(String(64), primary_key=True)
    chat_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("chats.id", ondelete="CASCADE"), index=True
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now_utc)


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

# SQLite ships with foreign-key enforcement OFF per connection; without this
# the ondelete="CASCADE"/"SET NULL" clauses above are silently ignored and
# deleting a chat leaves orphaned findings/shares behind.
if DATABASE_URL.startswith("sqlite"):
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _enable_sqlite_fks(dbapi_conn, _record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


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

    # Migration: dashboard_charts layout fields (Wave 1).
    if "dashboard_charts" in insp.get_table_names():
        dash_cols = {c["name"] for c in insp.get_columns("dashboard_charts")}
        with engine.begin() as conn:
            if "layout_w" not in dash_cols:
                conn.execute(text("ALTER TABLE dashboard_charts ADD COLUMN layout_w INTEGER NOT NULL DEFAULT 6"))
            if "layout_h" not in dash_cols:
                conn.execute(text("ALTER TABLE dashboard_charts ADD COLUMN layout_h INTEGER NOT NULL DEFAULT 2"))
            if "chart_kind" not in dash_cols:
                conn.execute(text("ALTER TABLE dashboard_charts ADD COLUMN chart_kind VARCHAR(32) NOT NULL DEFAULT 'plotly'"))
            if "kpi_json" not in dash_cols:
                conn.execute(text("ALTER TABLE dashboard_charts ADD COLUMN kpi_json TEXT"))
            if "finding_id" not in dash_cols:
                conn.execute(text(
                    "ALTER TABLE dashboard_charts ADD COLUMN finding_id VARCHAR(36) "
                    "REFERENCES findings(id) ON DELETE SET NULL"
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
