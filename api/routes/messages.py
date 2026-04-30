"""Message-level operations (pin/unpin)."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api import state
from api.auth import get_current_user_id

router = APIRouter(prefix="/messages", tags=["messages"])


class PatchMessageBody(BaseModel):
    pinned: Optional[bool] = None


@router.patch("/{message_id}")
async def patch_message(
    message_id: str,
    body: PatchMessageBody,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    if body.pinned is None:
        raise HTTPException(400, "Nothing to update")
    if not state.set_message_pinned(message_id, user_id, body.pinned):
        raise HTTPException(404, "Message not found")
    return {"id": message_id, "pinned": body.pinned}
