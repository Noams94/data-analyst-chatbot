"""Report export.

Emits a `ReportDoc` JSON document for a chat: a curated, publishable summary
of the conversation. Charts are inlined as base64 PNG so the document is
self-contained and can be POSTed to reports.noamkeshet.com (or downloaded
and rendered as standalone HTML).
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api import state

router = APIRouter(tags=["reports"])


def _data_url_from_path(path: Path) -> str:
    if not path.exists():
        return ""
    raw = path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"


@router.get("/chats/{chat_id}/report")
async def build_report(chat_id: str) -> dict:
    chat = state.get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    dataset = state.get_dataset(chat.dataset_id)

    # Build sections from pinned (user, assistant) pairs in order.
    sections: list[dict] = []
    pending_question: str | None = None
    for m in chat.messages:
        if m.role == "user":
            pending_question = m.content
            continue
        if m.role == "assistant" and m.pinned and m.content.strip():
            sections.append({
                "id": m.id,
                "question": pending_question or "",
                "bodyMd": m.content,
                "charts": [
                    {
                        "id": c.id,
                        "title": c.title,
                        "chartType": c.chart_type,
                        "dataUrl": _data_url_from_path(c.path),
                    }
                    for c in m.charts
                ],
                "snippets": [
                    {"id": sn.id, "type": sn.type, "code": sn.code}
                    for sn in m.snippets
                ],
                "createdAt": m.created_at,
            })
            pending_question = None

    return {
        "chatId": chat.id,
        "datasetName": dataset.name if dataset else "",
        "datasetMeta": {
            "rowCount": dataset.row_count if dataset else 0,
            "columns": dataset.columns if dataset else [],
        },
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sections": sections,
    }
