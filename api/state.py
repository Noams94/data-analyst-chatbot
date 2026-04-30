"""Persistence layer.

DB-backed via SQLAlchemy. DataFrames live as parquet files on disk and are
loaded on demand (LRU-cached). Charts live as PNG files and are served
through GET /charts/:id.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from api import db


def new_id() -> str:
    return str(uuid4())


# ─── DataFrame cache ─────────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def load_dataframe(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


# ─── Datasets ────────────────────────────────────────────────────────────────

@dataclass
class DatasetRecord:
    id: str
    name: str
    columns: list[str]
    row_count: int
    size_bytes: int
    parquet_path: str
    created_at: str


def _dataset_to_record(d: db.Dataset) -> DatasetRecord:
    return DatasetRecord(
        id=d.id,
        name=d.name,
        columns=list(d.columns or []),
        row_count=d.row_count,
        size_bytes=d.size_bytes,
        parquet_path=d.parquet_path,
        created_at=d.created_at.isoformat(),
    )


def create_dataset(name: str, df: pd.DataFrame, size_bytes: int) -> DatasetRecord:
    dataset_id = new_id()
    parquet_path = str(db.DATASETS_DIR / f"{dataset_id}.parquet")
    df.to_parquet(parquet_path, index=False)
    with db.get_session() as s:
        row = db.Dataset(
            id=dataset_id,
            name=name,
            parquet_path=parquet_path,
            columns=df.columns.astype(str).tolist(),
            row_count=int(df.shape[0]),
            size_bytes=size_bytes,
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        return _dataset_to_record(row)


def get_dataset(dataset_id: str) -> DatasetRecord | None:
    with db.get_session() as s:
        row = s.get(db.Dataset, dataset_id)
        return _dataset_to_record(row) if row else None


def get_dataset_dataframe(dataset_id: str) -> pd.DataFrame | None:
    rec = get_dataset(dataset_id)
    if not rec:
        return None
    return load_dataframe(rec.parquet_path)


# ─── Chats + messages ────────────────────────────────────────────────────────

@dataclass
class ChartRecord:
    id: str
    message_id: str
    path: Path
    title: str
    chart_type: str


@dataclass
class SnippetRecord:
    id: str
    message_id: str
    type: str
    code: str


@dataclass
class MessageRecord:
    id: str
    chat_id: str
    role: str
    content: str
    charts: list[ChartRecord]
    snippets: list[SnippetRecord]
    created_at: str


@dataclass
class ChatRecord:
    id: str
    dataset_id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[MessageRecord]


def _chart_to_record(c: db.Chart) -> ChartRecord:
    return ChartRecord(
        id=c.id, message_id=c.message_id, path=Path(c.path),
        title=c.title, chart_type=c.chart_type,
    )


def _snippet_to_record(sn: db.Snippet) -> SnippetRecord:
    return SnippetRecord(id=sn.id, message_id=sn.message_id, type=sn.type, code=sn.code)


def _message_to_record(m: db.Message) -> MessageRecord:
    return MessageRecord(
        id=m.id, chat_id=m.chat_id, role=m.role, content=m.content,
        charts=[_chart_to_record(c) for c in m.charts],
        snippets=[_snippet_to_record(sn) for sn in m.snippets],
        created_at=m.created_at.isoformat(),
    )


def _chat_to_record(c: db.Chat) -> ChatRecord:
    return ChatRecord(
        id=c.id, dataset_id=c.dataset_id, title=c.title,
        created_at=c.created_at.isoformat(),
        updated_at=c.updated_at.isoformat(),
        messages=[_message_to_record(m) for m in c.messages],
    )


def create_chat(dataset_id: str, title: str = "New chat") -> ChatRecord | None:
    with db.get_session() as s:
        if not s.get(db.Dataset, dataset_id):
            return None
        chat = db.Chat(id=new_id(), dataset_id=dataset_id, title=title)
        s.add(chat)
        s.commit()
        s.refresh(chat)
        return _chat_to_record(chat)


def get_chat(chat_id: str) -> ChatRecord | None:
    with db.get_session() as s:
        chat = s.get(db.Chat, chat_id)
        return _chat_to_record(chat) if chat else None


def add_user_message(chat_id: str, content: str) -> MessageRecord:
    with db.get_session() as s:
        m = db.Message(id=new_id(), chat_id=chat_id, role="user", content=content)
        s.add(m)
        s.commit()
        s.refresh(m)
        return _message_to_record(m)


def create_assistant_placeholder(chat_id: str) -> str:
    """Insert an empty assistant message and return its id."""
    msg_id = new_id()
    with db.get_session() as s:
        s.add(db.Message(id=msg_id, chat_id=chat_id, role="assistant", content=""))
        s.commit()
    return msg_id


def append_chart(message_id: str, source_path: Path, title: str, chart_type: str) -> ChartRecord:
    chart_id = new_id()
    final_path = db.CHARTS_DIR / f"{chart_id}.png"
    # Move the per-request tempfile into the persistent charts directory.
    source_path.replace(final_path)
    with db.get_session() as s:
        c = db.Chart(
            id=chart_id, message_id=message_id, path=str(final_path),
            title=title, chart_type=chart_type,
        )
        s.add(c)
        s.commit()
    return ChartRecord(id=chart_id, message_id=message_id, path=final_path,
                       title=title, chart_type=chart_type)


def append_snippet(message_id: str, type_: str, code: str) -> SnippetRecord:
    snippet_id = new_id()
    with db.get_session() as s:
        sn = db.Snippet(id=snippet_id, message_id=message_id, type=type_, code=code)
        s.add(sn)
        s.commit()
    return SnippetRecord(id=snippet_id, message_id=message_id, type=type_, code=code)


def finalize_assistant_message(message_id: str, content: str) -> None:
    """Persist the streamed text into the assistant message row."""
    with db.get_session() as s:
        m = s.get(db.Message, message_id)
        if m:
            m.content = content
            s.commit()


def get_chart(chart_id: str) -> ChartRecord | None:
    with db.get_session() as s:
        c = s.get(db.Chart, chart_id)
        return _chart_to_record(c) if c else None


# ─── Wire-format helpers ─────────────────────────────────────────────────────

def message_to_dict(m: MessageRecord) -> dict[str, Any]:
    return {
        "id": m.id,
        "chatId": m.chat_id,
        "role": m.role,
        "content": m.content,
        "charts": [
            {"id": c.id, "url": f"/charts/{c.id}", "title": c.title, "chartType": c.chart_type}
            for c in m.charts
        ],
        "snippets": [
            {"id": s.id, "type": s.type, "code": s.code} for s in m.snippets
        ],
        "createdAt": m.created_at,
    }
