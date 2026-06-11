"""Findings + dashboard-layout routes (Workspace Wave 1)."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api import state
from api.auth import get_current_user_id

router = APIRouter(tags=["workspace"])


# ─── Findings ────────────────────────────────────────────────────────────────


class PatchFindingBody(BaseModel):
    title: Optional[str] = None
    claim: Optional[str] = None
    confidence: Optional[str] = None
    dimension: Optional[str] = None
    includedInReport: Optional[bool] = None
    includedInDashboard: Optional[bool] = None
    position: Optional[int] = None


class ReorderBody(BaseModel):
    orderedIds: List[str]


@router.get("/chats/{chat_id}/findings")
async def list_findings(
    chat_id: str,
    user_id: str = Depends(get_current_user_id),
) -> list[dict]:
    rows = state.list_findings(chat_id, user_id)
    if rows is None:
        raise HTTPException(404, "Chat not found")
    return [state.finding_to_dict(f) for f in rows]


@router.patch("/findings/{finding_id}")
async def patch_finding(
    finding_id: str,
    body: PatchFindingBody,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    updated = state.patch_finding(
        finding_id, user_id,
        title=body.title, claim=body.claim,
        confidence=body.confidence, dimension=body.dimension,
        included_in_report=body.includedInReport,
        included_in_dashboard=body.includedInDashboard,
        position=body.position,
    )
    if not updated:
        raise HTTPException(404, "Finding not found")
    # Keep the dashboard in sync with the flag: turning "in dashboard" off
    # also removes the chart(s) that were promoted from this finding.
    if body.includedInDashboard is False:
        state.delete_dashboard_charts_for_finding(updated.chat_id, user_id, finding_id)
    return state.finding_to_dict(updated)


@router.delete("/findings/{finding_id}", status_code=204, response_model=None)
async def delete_finding(
    finding_id: str,
    user_id: str = Depends(get_current_user_id),
) -> None:
    if not state.delete_finding(finding_id, user_id):
        raise HTTPException(404, "Finding not found")


@router.post("/chats/{chat_id}/findings/reorder")
async def reorder_findings(
    chat_id: str,
    body: ReorderBody,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    if not state.reorder_findings(chat_id, user_id, body.orderedIds):
        raise HTTPException(404, "Chat not found")
    return {"ok": True}


@router.post("/chats/{chat_id}/findings/{finding_id}/promote-to-dashboard")
async def promote_to_dashboard(
    chat_id: str,
    finding_id: str,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    finding = state.get_finding(finding_id, user_id)
    if not finding or finding.chat_id != chat_id:
        raise HTTPException(404, "Finding not found")
    if not finding.chart_id:
        raise HTTPException(400, "Finding has no attached chart")
    existing = state.get_dashboard(chat_id)
    # Idempotent: a second promote returns the chart already on the dashboard.
    dup = next((c for c in existing if c.finding_id == finding_id), None)
    if dup:
        state.patch_finding(finding_id, user_id, included_in_dashboard=True)
        return state.dashboard_chart_to_dict(dup)
    spec = state.get_plotly_chart_spec(finding.chart_id)
    if not spec:
        raise HTTPException(404, "Chart not found")
    spec_json, chart_title = spec
    title = finding.title or chart_title or "Finding"
    rec = state.append_dashboard_chart(
        chat_id, spec_json, title, position=len(existing),
        finding_id=finding_id,
    )
    state.patch_finding(finding_id, user_id, included_in_dashboard=True)
    return state.dashboard_chart_to_dict(rec)


# ─── Dashboard layout ────────────────────────────────────────────────────────


class PatchDashboardBody(BaseModel):
    title: Optional[str] = None
    layoutW: Optional[int] = None
    layoutH: Optional[int] = None
    position: Optional[int] = None


@router.patch("/chats/{chat_id}/dashboard/{chart_id}")
async def patch_dashboard_chart(
    chat_id: str,
    chart_id: str,
    body: PatchDashboardBody,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    updated = state.update_dashboard_chart(
        chat_id, user_id, chart_id,
        title=body.title, layout_w=body.layoutW,
        layout_h=body.layoutH, position=body.position,
    )
    if not updated:
        raise HTTPException(404, "Dashboard chart not found")
    return state.dashboard_chart_to_dict(updated)


@router.delete("/chats/{chat_id}/dashboard/{chart_id}", status_code=204, response_model=None)
async def delete_dashboard_chart(
    chat_id: str,
    chart_id: str,
    user_id: str = Depends(get_current_user_id),
) -> None:
    if not state.delete_dashboard_chart(chat_id, user_id, chart_id):
        raise HTTPException(404, "Dashboard chart not found")


@router.post("/chats/{chat_id}/dashboard/reorder")
async def reorder_dashboard(
    chat_id: str,
    body: ReorderBody,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    if not state.reorder_dashboard(chat_id, user_id, body.orderedIds):
        raise HTTPException(404, "Chat not found")
    return {"ok": True}
