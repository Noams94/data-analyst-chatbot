"""Report export.

Emits a `ReportDoc` JSON document for a chat: a curated, publishable summary
of the conversation. Charts are inlined as base64 PNG so the document is
self-contained and can be POSTed to reports.noamkeshet.com (or downloaded
and rendered as standalone HTML).
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from api import state
from api.auth import get_current_user_id

router = APIRouter(tags=["reports"])


def _data_url_from_bytes(data: bytes) -> str:
    if not data:
        return ""
    return f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"


@router.get("/chats/{chat_id}/report")
async def build_report(
    chat_id: str,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    chat = state.get_chat(chat_id, user_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    dataset = state.get_dataset(chat.dataset_id, user_id)

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
                        "dataUrl": _data_url_from_bytes(c.image_bytes),
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
