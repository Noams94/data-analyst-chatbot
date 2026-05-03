"""User provider/model settings — GET and PATCH."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api import state
from api.auth import get_current_user_id

router = APIRouter(tags=["settings"])


def _masked(key: str) -> str:
    """Return a safe display string for a secret key."""
    if not key:
        return ""
    if len(key) <= 8:
        return "***"
    return key[:4] + "***" + key[-4:]


def _to_wire(settings: dict) -> dict:
    return {
        "provider": settings.get("provider", "anthropic"),
        "anthropicModel": settings.get("anthropic_model", ""),
        "anthropicApiKey": _masked(settings.get("anthropic_api_key", "")),
        "hasAnthropicKey": bool(settings.get("anthropic_api_key")),
        "ollamaModel": settings.get("ollama_model", ""),
        "ollamaBaseUrl": settings.get("ollama_base_url", ""),
    }


@router.get("/settings")
async def get_settings(user_id: str = Depends(get_current_user_id)) -> dict:
    return _to_wire(state.get_user_settings(user_id))


class PatchSettingsBody(BaseModel):
    provider: Optional[str] = None
    anthropicModel: Optional[str] = None
    anthropicApiKey: Optional[str] = None
    ollamaModel: Optional[str] = None
    ollamaBaseUrl: Optional[str] = None


@router.patch("/settings")
async def patch_settings(
    body: PatchSettingsBody,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    updates: dict = {}
    if body.provider is not None:
        updates["provider"] = body.provider
    if body.anthropicModel is not None:
        updates["anthropic_model"] = body.anthropicModel
    if body.anthropicApiKey is not None:
        updates["anthropic_api_key"] = body.anthropicApiKey
    if body.ollamaModel is not None:
        updates["ollama_model"] = body.ollamaModel
    if body.ollamaBaseUrl is not None:
        updates["ollama_base_url"] = body.ollamaBaseUrl
    new_settings = state.update_user_settings(user_id, updates)
    return _to_wire(new_settings)
