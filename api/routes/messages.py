"""Message-level operations (pin/unpin)."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api import state

router = APIRouter(prefix="/messages", tags=["messages"])


class PatchMessageBody(BaseModel):
    pinned: Optional[bool] = None


@router.patch("/{message_id}")
async def patch_message(message_id: str, body: PatchMessageBody) -> dict:
    if body.pinned is None:
        raise HTTPException(400, "Nothing to update")
    if not state.set_message_pinned(message_id, body.pinned):
        raise HTTPException(404, "Message not found")
    return {"id": message_id, "pinned": body.pinned}
