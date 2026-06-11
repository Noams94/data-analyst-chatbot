"""Persistence layer.

DB-backed via SQLAlchemy. DataFrames live as parquet files on disk and are
loaded on demand (LRU-cached). Charts live as PNG files and are served
through GET /charts/:id.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
    user_id: str
    name: str
    columns: list[str]
    row_count: int
    size_bytes: int
    parquet_path: str
    created_at: str


def _dataset_to_record(d: db.Dataset) -> DatasetRecord:
    return DatasetRecord(
        id=d.id,
        user_id=d.user_id,
        name=d.name,
        columns=list(d.columns or []),
        row_count=d.row_count,
        size_bytes=d.size_bytes,
        parquet_path=d.parquet_path,
        created_at=d.created_at.isoformat(),
    )


def create_dataset(user_id: str, name: str, df: pd.DataFrame, size_bytes: int) -> DatasetRecord:
    dataset_id = new_id()
    parquet_path = str(db.DATASETS_DIR / f"{dataset_id}.parquet")
    df.to_parquet(parquet_path, index=False)
    with db.get_session() as s:
        row = db.Dataset(
            id=dataset_id,
            user_id=user_id,
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


def get_dataset(dataset_id: str, user_id: str) -> DatasetRecord | None:
    with db.get_session() as s:
        row = s.get(db.Dataset, dataset_id)
        if row and row.user_id == user_id:
            return _dataset_to_record(row)
        return None


def list_datasets(user_id: str) -> list[DatasetRecord]:
    """All datasets owned by `user_id`, newest first."""
    from sqlalchemy import select
    with db.get_session() as s:
        rows = s.execute(
            select(db.Dataset)
            .where(db.Dataset.user_id == user_id)
            .order_by(db.Dataset.created_at.desc())
        ).scalars().all()
        return [_dataset_to_record(r) for r in rows]


def delete_dataset(dataset_id: str, user_id: str) -> bool:
    with db.get_session() as s:
        row = s.get(db.Dataset, dataset_id)
        if not row or row.user_id != user_id:
            return False
        try:
            Path(row.parquet_path).unlink(missing_ok=True)
        except OSError:
            pass
        s.delete(row)
        s.commit()
        return True


def delete_chat(chat_id: str, user_id: str) -> bool:
    with db.get_session() as s:
        row = s.get(db.Chat, chat_id)
        if not row or row.user_id != user_id:
            return False
        s.delete(row)
        s.commit()
        return True


def update_chat_title(chat_id: str, title: str) -> None:
    with db.get_session() as s:
        row = s.get(db.Chat, chat_id)
        if row:
            row.title = title
            s.commit()


def get_dataset_dataframe(dataset_id: str, user_id: str) -> pd.DataFrame | None:
    rec = get_dataset(dataset_id, user_id)
    if not rec:
        return None
    return load_dataframe(rec.parquet_path)


# ─── Chats + messages ────────────────────────────────────────────────────────

@dataclass
class PlotlyChartRecord:
    id: str
    message_id: str
    spec_json: str
    title: str


@dataclass
class ChartRecord:
    id: str
    message_id: str
    image_bytes: bytes
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
    plotly_charts: list[PlotlyChartRecord]
    created_at: str
    pinned: bool = True


@dataclass
class ChatRecord:
    id: str
    user_id: str
    dataset_id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[MessageRecord]


def _chart_to_record(c: db.Chart) -> ChartRecord:
    return ChartRecord(
        id=c.id, message_id=c.message_id,
        image_bytes=c.image_bytes or b"",
        title=c.title, chart_type=c.chart_type,
    )


def _snippet_to_record(sn: db.Snippet) -> SnippetRecord:
    return SnippetRecord(id=sn.id, message_id=sn.message_id, type=sn.type, code=sn.code)


def _plotly_chart_to_record(c: db.PlotlyChart) -> PlotlyChartRecord:
    return PlotlyChartRecord(id=c.id, message_id=c.message_id, spec_json=c.spec_json, title=c.title)


def _message_to_record(m: db.Message) -> MessageRecord:
    return MessageRecord(
        id=m.id, chat_id=m.chat_id, role=m.role, content=m.content,
        charts=[_chart_to_record(c) for c in m.charts],
        snippets=[_snippet_to_record(sn) for sn in m.snippets],
        plotly_charts=[_plotly_chart_to_record(pc) for pc in m.plotly_charts],
        created_at=m.created_at.isoformat(),
        pinned=bool(getattr(m, "pinned", True)),
    )


def _chat_to_record(c: db.Chat) -> ChatRecord:
    return ChatRecord(
        id=c.id, user_id=c.user_id, dataset_id=c.dataset_id, title=c.title,
        created_at=c.created_at.isoformat(),
        updated_at=c.updated_at.isoformat(),
        messages=[_message_to_record(m) for m in c.messages],
    )


def create_chat(user_id: str, dataset_id: str, title: str = "New chat") -> ChatRecord | None:
    with db.get_session() as s:
        ds = s.get(db.Dataset, dataset_id)
        if not ds or ds.user_id != user_id:
            return None
        chat = db.Chat(id=new_id(), user_id=user_id, dataset_id=dataset_id, title=title)
        s.add(chat)
        s.commit()
        s.refresh(chat)
        return _chat_to_record(chat)


def get_chat(chat_id: str, user_id: str) -> ChatRecord | None:
    with db.get_session() as s:
        chat = s.get(db.Chat, chat_id)
        if not chat or chat.user_id != user_id:
            return None
        # Remove empty assistant placeholders left by interrupted streams.
        # Skip placeholders younger than 60s — those are in-flight streams whose
        # creator just called us to fetch history.
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=60)
        def _is_old(m_created):
            if m_created.tzinfo is None:
                m_created = m_created.replace(tzinfo=timezone.utc)
            return m_created < cutoff
        stale = [
            m for m in chat.messages
            if m.role == "assistant" and not m.content.strip()
            and not m.charts and not m.snippets
            and _is_old(m.created_at)
        ]
        for m in stale:
            s.delete(m)
        if stale:
            s.commit()
            s.refresh(chat)
        return _chat_to_record(chat)


@dataclass
class ChatSummary:
    id: str
    dataset_id: str
    dataset_name: str
    title: str
    message_count: int
    last_message_at: str | None
    created_at: str
    updated_at: str


def list_chats(user_id: str, dataset_id: str | None = None) -> list[ChatSummary]:
    """List chats owned by `user_id`, newest first, optionally filtered by
    dataset. Each summary carries the dataset name + message count + last
    message timestamp so the sidebar can render in one round-trip."""
    from sqlalchemy import func, select
    with db.get_session() as s:
        last_msg_subq = (
            select(
                db.Message.chat_id.label("chat_id"),
                func.count(db.Message.id).label("msg_count"),
                func.max(db.Message.created_at).label("last_at"),
            )
            .group_by(db.Message.chat_id)
            .subquery()
        )
        q = (
            select(
                db.Chat,
                db.Dataset.name.label("dataset_name"),
                last_msg_subq.c.msg_count,
                last_msg_subq.c.last_at,
            )
            .join(db.Dataset, db.Dataset.id == db.Chat.dataset_id)
            .join(last_msg_subq, last_msg_subq.c.chat_id == db.Chat.id, isouter=True)
            .where(db.Chat.user_id == user_id)
            .order_by(db.Chat.updated_at.desc())
        )
        if dataset_id is not None:
            q = q.where(db.Chat.dataset_id == dataset_id)
        rows = s.execute(q).all()
        out: list[ChatSummary] = []
        for chat, ds_name, count, last_at in rows:
            out.append(ChatSummary(
                id=chat.id,
                dataset_id=chat.dataset_id,
                dataset_name=ds_name or "",
                title=chat.title,
                message_count=int(count or 0),
                last_message_at=last_at.isoformat() if last_at else None,
                created_at=chat.created_at.isoformat(),
                updated_at=chat.updated_at.isoformat(),
            ))
        return out


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
    """Read the per-request tempfile, persist the PNG bytes in the DB, then
    delete the tempfile. Cloud-friendly: no persistent filesystem needed.
    """
    chart_id = new_id()
    image_bytes = source_path.read_bytes()
    try:
        source_path.unlink()
    except OSError:
        pass
    with db.get_session() as s:
        c = db.Chart(
            id=chart_id, message_id=message_id,
            image_bytes=image_bytes,
            title=title, chart_type=chart_type,
        )
        s.add(c)
        s.commit()
    return ChartRecord(id=chart_id, message_id=message_id,
                       image_bytes=image_bytes,
                       title=title, chart_type=chart_type)


def append_plotly_chart(message_id: str, spec_json: str, title: str) -> PlotlyChartRecord:
    chart_id = new_id()
    with db.get_session() as s:
        c = db.PlotlyChart(id=chart_id, message_id=message_id, spec_json=spec_json, title=title)
        s.add(c)
        s.commit()
    return PlotlyChartRecord(id=chart_id, message_id=message_id, spec_json=spec_json, title=title)


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


def get_chart(chart_id: str, user_id: str) -> ChartRecord | None:
    """Returns the chart only if it belongs to a chat owned by `user_id`."""
    with db.get_session() as s:
        c = s.get(db.Chart, chart_id)
        if not c:
            return None
        msg = s.get(db.Message, c.message_id)
        if not msg:
            return None
        chat = s.get(db.Chat, msg.chat_id)
        if not chat or chat.user_id != user_id:
            return None
        return _chart_to_record(c)


def set_message_pinned(message_id: str, user_id: str, pinned: bool) -> bool:
    """Toggle pin flag, scoped by ownership. Returns True if the message
    existed AND is owned by `user_id`."""
    with db.get_session() as s:
        m = s.get(db.Message, message_id)
        if not m:
            return False
        chat = s.get(db.Chat, m.chat_id)
        if not chat or chat.user_id != user_id:
            return False
        m.pinned = pinned
        s.commit()
        return True


def delete_message(message_id: str, user_id: str) -> bool:
    with db.get_session() as s:
        m = s.get(db.Message, message_id)
        if not m:
            return False
        chat = s.get(db.Chat, m.chat_id)
        if not chat or chat.user_id != user_id:
            return False
        s.delete(m)
        s.commit()
        return True


# ─── User settings ───────────────────────────────────────────────────────────

import json as _json
import os as _os

_SETTINGS_DEFAULTS = {
    "provider": _os.getenv("PROVIDER", "anthropic" if _os.getenv("ANTHROPIC_API_KEY") else "ollama"),
    "anthropic_model": _os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
    "anthropic_api_key": _os.getenv("ANTHROPIC_API_KEY", ""),
    "ollama_model": _os.getenv("OLLAMA_MODEL", "llama3.1"),
    "ollama_base_url": _os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
}

_ALLOWED_KEYS = set(_SETTINGS_DEFAULTS)


def get_user_settings(user_id: str) -> dict:
    """Return merged settings: DB values take precedence over env-var defaults."""
    with db.get_session() as s:
        row = s.get(db.UserSettings, user_id)
        stored: dict = _json.loads(row.settings_json) if row else {}
    merged = dict(_SETTINGS_DEFAULTS)
    merged.update({k: v for k, v in stored.items() if k in _ALLOWED_KEYS})
    return merged


def update_user_settings(user_id: str, updates: dict) -> dict:
    """Persist whitelisted keys; return new merged settings."""
    safe = {k: v for k, v in updates.items() if k in _ALLOWED_KEYS}
    with db.get_session() as s:
        row = s.get(db.UserSettings, user_id)
        if row is None:
            row = db.UserSettings(user_id=user_id, settings_json=_json.dumps(safe))
            s.add(row)
        else:
            existing = _json.loads(row.settings_json)
            existing.update(safe)
            row.settings_json = _json.dumps(existing)
            row.updated_at = db._now_utc()
        s.commit()
    return get_user_settings(user_id)


# ─── Dashboard ───────────────────────────────────────────────────────────────

@dataclass
class DashboardChartRecord:
    id: str
    chat_id: str
    spec_json: str
    title: str
    position: int
    layout_w: int = 6
    layout_h: int = 2
    chart_kind: str = "plotly"
    kpi_json: str | None = None
    finding_id: str | None = None


def append_dashboard_chart(
    chat_id: str, spec_json: str, title: str, position: int,
    layout_w: int = 6, layout_h: int = 2,
    chart_kind: str = "plotly", kpi_json: str | None = None,
    finding_id: str | None = None,
) -> DashboardChartRecord:
    chart_id = new_id()
    with db.get_session() as s:
        row = db.DashboardChart(
            id=chart_id, chat_id=chat_id,
            spec_json=spec_json, title=title, position=position,
            layout_w=layout_w, layout_h=layout_h,
            chart_kind=chart_kind, kpi_json=kpi_json,
            finding_id=finding_id,
        )
        s.add(row)
        s.commit()
    return DashboardChartRecord(
        id=chart_id, chat_id=chat_id,
        spec_json=spec_json, title=title, position=position,
        layout_w=layout_w, layout_h=layout_h,
        chart_kind=chart_kind, kpi_json=kpi_json,
        finding_id=finding_id,
    )


def _dashboard_to_record(r: db.DashboardChart) -> DashboardChartRecord:
    return DashboardChartRecord(
        id=r.id, chat_id=r.chat_id,
        spec_json=r.spec_json, title=r.title, position=r.position,
        layout_w=getattr(r, "layout_w", 6) or 6,
        layout_h=getattr(r, "layout_h", 2) or 2,
        chart_kind=getattr(r, "chart_kind", "plotly") or "plotly",
        kpi_json=getattr(r, "kpi_json", None),
        finding_id=getattr(r, "finding_id", None),
    )


def get_dashboard(chat_id: str) -> list[DashboardChartRecord]:
    from sqlalchemy import select
    with db.get_session() as s:
        rows = s.execute(
            select(db.DashboardChart)
            .where(db.DashboardChart.chat_id == chat_id)
            .order_by(db.DashboardChart.position)
        ).scalars().all()
        return [_dashboard_to_record(r) for r in rows]


def update_dashboard_chart(
    chat_id: str, user_id: str, chart_id: str,
    title: str | None = None,
    layout_w: int | None = None,
    layout_h: int | None = None,
    position: int | None = None,
) -> DashboardChartRecord | None:
    with db.get_session() as s:
        chat = s.get(db.Chat, chat_id)
        if not chat or chat.user_id != user_id:
            return None
        row = s.get(db.DashboardChart, chart_id)
        if not row or row.chat_id != chat_id:
            return None
        if title is not None:
            row.title = title
        if layout_w is not None:
            row.layout_w = max(1, min(12, int(layout_w)))
        if layout_h is not None:
            row.layout_h = max(1, min(6, int(layout_h)))
        if position is not None:
            row.position = int(position)
        s.commit()
        s.refresh(row)
        return _dashboard_to_record(row)


def reorder_dashboard(chat_id: str, user_id: str, ordered_ids: list[str]) -> bool:
    with db.get_session() as s:
        chat = s.get(db.Chat, chat_id)
        if not chat or chat.user_id != user_id:
            return False
        for pos, cid in enumerate(ordered_ids):
            row = s.get(db.DashboardChart, cid)
            if row and row.chat_id == chat_id:
                row.position = pos
        s.commit()
        return True


def delete_dashboard_charts_for_finding(
    chat_id: str, user_id: str, finding_id: str
) -> list[str]:
    """Remove dashboard charts that were promoted from this finding.
    Returns the deleted chart ids."""
    from sqlalchemy import select
    with db.get_session() as s:
        chat = s.get(db.Chat, chat_id)
        if not chat or chat.user_id != user_id:
            return []
        rows = s.execute(
            select(db.DashboardChart)
            .where(db.DashboardChart.chat_id == chat_id)
            .where(db.DashboardChart.finding_id == finding_id)
        ).scalars().all()
        deleted = [r.id for r in rows]
        for r in rows:
            s.delete(r)
        if rows:
            s.commit()
        return deleted


def delete_dashboard_chart(chat_id: str, user_id: str, chart_id: str) -> bool:
    with db.get_session() as s:
        chat = s.get(db.Chat, chat_id)
        if not chat or chat.user_id != user_id:
            return False
        row = s.get(db.DashboardChart, chart_id)
        if not row or row.chat_id != chat_id:
            return False
        s.delete(row)
        s.commit()
        return True


def clear_dashboard(chat_id: str) -> int:
    from sqlalchemy import delete
    with db.get_session() as s:
        result = s.execute(
            delete(db.DashboardChart).where(db.DashboardChart.chat_id == chat_id)
        )
        s.commit()
        return result.rowcount


def dashboard_chart_to_dict(r: DashboardChartRecord) -> dict:
    return {
        "id": r.id,
        "chatId": r.chat_id,
        "spec": r.spec_json,
        "title": r.title,
        "position": r.position,
        "layoutW": r.layout_w,
        "layoutH": r.layout_h,
        "chartKind": r.chart_kind,
        "kpi": _json.loads(r.kpi_json) if r.kpi_json else None,
        "findingId": r.finding_id,
    }


# ─── Findings ────────────────────────────────────────────────────────────────

@dataclass
class FindingRecord:
    id: str
    chat_id: str
    message_id: str | None
    chart_id: str | None
    title: str
    claim: str
    evidence: dict
    confidence: str
    dimension: str
    position: int
    included_in_report: bool
    included_in_dashboard: bool
    created_at: str


def _finding_to_record(f: db.Finding) -> FindingRecord:
    try:
        evidence = _json.loads(f.evidence_json) if f.evidence_json else {}
    except Exception:
        evidence = {}
    return FindingRecord(
        id=f.id, chat_id=f.chat_id, message_id=f.message_id, chart_id=f.chart_id,
        title=f.title, claim=f.claim, evidence=evidence,
        confidence=f.confidence, dimension=f.dimension,
        position=f.position,
        included_in_report=bool(f.included_in_report),
        included_in_dashboard=bool(f.included_in_dashboard),
        created_at=f.created_at.isoformat(),
    )


def append_finding(
    chat_id: str, message_id: str | None, title: str, claim: str,
    evidence: dict, confidence: str = "medium", dimension: str = "",
    chart_id: str | None = None,
) -> FindingRecord:
    from sqlalchemy import func, select
    finding_id = new_id()
    with db.get_session() as s:
        next_pos = s.execute(
            select(func.coalesce(func.max(db.Finding.position), -1) + 1)
            .where(db.Finding.chat_id == chat_id)
        ).scalar_one()
        row = db.Finding(
            id=finding_id, chat_id=chat_id, message_id=message_id,
            chart_id=chart_id, title=title[:512], claim=claim,
            evidence_json=_json.dumps(evidence, default=str),
            confidence=confidence if confidence in ("low", "medium", "high") else "medium",
            dimension=dimension[:64], position=int(next_pos),
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        return _finding_to_record(row)


def list_findings(chat_id: str, user_id: str) -> list[FindingRecord] | None:
    from sqlalchemy import select
    with db.get_session() as s:
        chat = s.get(db.Chat, chat_id)
        if not chat or chat.user_id != user_id:
            return None
        rows = s.execute(
            select(db.Finding)
            .where(db.Finding.chat_id == chat_id)
            .order_by(db.Finding.position, db.Finding.created_at)
        ).scalars().all()
        return [_finding_to_record(r) for r in rows]


def patch_finding(
    finding_id: str, user_id: str, *,
    title: str | None = None, claim: str | None = None,
    confidence: str | None = None, dimension: str | None = None,
    included_in_report: bool | None = None,
    included_in_dashboard: bool | None = None,
    position: int | None = None,
) -> FindingRecord | None:
    with db.get_session() as s:
        row = s.get(db.Finding, finding_id)
        if not row:
            return None
        chat = s.get(db.Chat, row.chat_id)
        if not chat or chat.user_id != user_id:
            return None
        if title is not None:
            row.title = title[:512]
        if claim is not None:
            row.claim = claim
        if confidence is not None and confidence in ("low", "medium", "high"):
            row.confidence = confidence
        if dimension is not None:
            row.dimension = dimension[:64]
        if included_in_report is not None:
            row.included_in_report = bool(included_in_report)
        if included_in_dashboard is not None:
            row.included_in_dashboard = bool(included_in_dashboard)
        if position is not None:
            row.position = int(position)
        s.commit()
        s.refresh(row)
        return _finding_to_record(row)


def delete_finding(finding_id: str, user_id: str) -> bool:
    with db.get_session() as s:
        row = s.get(db.Finding, finding_id)
        if not row:
            return False
        chat = s.get(db.Chat, row.chat_id)
        if not chat or chat.user_id != user_id:
            return False
        s.delete(row)
        s.commit()
        return True


def reorder_findings(chat_id: str, user_id: str, ordered_ids: list[str]) -> bool:
    with db.get_session() as s:
        chat = s.get(db.Chat, chat_id)
        if not chat or chat.user_id != user_id:
            return False
        for pos, fid in enumerate(ordered_ids):
            row = s.get(db.Finding, fid)
            if row and row.chat_id == chat_id:
                row.position = pos
        s.commit()
        return True


def get_finding(finding_id: str, user_id: str) -> FindingRecord | None:
    with db.get_session() as s:
        row = s.get(db.Finding, finding_id)
        if not row:
            return None
        chat = s.get(db.Chat, row.chat_id)
        if not chat or chat.user_id != user_id:
            return None
        return _finding_to_record(row)


def finding_to_dict(f: FindingRecord) -> dict:
    return {
        "id": f.id,
        "chatId": f.chat_id,
        "messageId": f.message_id,
        "chartId": f.chart_id,
        "title": f.title,
        "claim": f.claim,
        "evidence": f.evidence,
        "confidence": f.confidence,
        "dimension": f.dimension,
        "position": f.position,
        "includedInReport": f.included_in_report,
        "includedInDashboard": f.included_in_dashboard,
        "createdAt": f.created_at,
    }


def get_plotly_chart_spec(chart_id: str) -> tuple[str, str] | None:
    """Return (spec_json, title) for a plotly chart, or None."""
    with db.get_session() as s:
        c = s.get(db.PlotlyChart, chart_id)
        if not c:
            return None
        return (c.spec_json, c.title)


# ─── Wire-format helpers ─────────────────────────────────────────────────────

def _data_url(image_bytes: bytes) -> str:
    if not image_bytes:
        return ""
    import base64
    return f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}"


def message_to_dict(m: MessageRecord) -> dict[str, Any]:
    return {
        "id": m.id,
        "chatId": m.chat_id,
        "role": m.role,
        "content": m.content,
        "pinned": m.pinned,
        "charts": [
            {
                "id": c.id,
                "dataUrl": _data_url(c.image_bytes),
                "title": c.title,
                "chartType": c.chart_type,
            }
            for c in m.charts
        ],
        "snippets": [
            {"id": s.id, "type": s.type, "code": s.code} for s in m.snippets
        ],
        "plotlyCharts": [
            {"id": pc.id, "spec": pc.spec_json, "title": pc.title}
            for pc in m.plotly_charts
        ],
        "createdAt": m.created_at,
    }
