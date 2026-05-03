"""Chat creation + streaming message route (SSE)."""

from __future__ import annotations

import asyncio
import os
import traceback
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api import state, tools
from api.auth import get_current_user_id
from api.chat_session import ChatSession, set_session
from api.prompts import SYSTEM_PROMPT_EN
from api.sse import sse

router = APIRouter(tags=["chats"])


def _history_turns(chat_record: state.ChatRecord) -> list:
    """Convert prior persisted user/assistant turns into chatlas Turn objects.
    Excludes the just-added user prompt and the empty assistant placeholder
    (the last two messages on the chat).
    """
    from chatlas import Turn
    turns = []
    for m in chat_record.messages[:-2]:
        if m.role in ("user", "assistant") and m.content.strip():
            turns.append(Turn(role=m.role, contents=m.content))
    return turns


def _generate_suggestions(user_msg: str, assistant_msg: str, columns: list, cfg: dict) -> list:
    """Return up to 3 short follow-up question strings, or [] on failure."""
    try:
        if cfg["provider"] == "anthropic":
            import anthropic, json as _json
            api_key = cfg.get("anthropic_api_key") or None
            client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
            cols_str = ", ".join(str(c) for c in columns[:20])
            resp = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=160,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Dataset columns: {cols_str}\n"
                        f"User asked: {user_msg[:200]}\n"
                        f"Assistant replied: {assistant_msg[:400]}\n\n"
                        "Suggest exactly 3 short follow-up questions the analyst might ask next. "
                        "Each must be under 10 words. "
                        "Return ONLY a valid JSON array of 3 strings, nothing else."
                    ),
                }],
            )
            text = resp.content[0].text.strip()
            start, end = text.index("["), text.rindex("]") + 1
            items = _json.loads(text[start:end])
            if isinstance(items, list):
                return [str(q) for q in items[:3] if q]
    except Exception:
        pass
    return []


def _generate_title(user_msg: str, assistant_msg: str, cfg: dict) -> str:
    """Generate a short chat title from the first exchange. Returns a plain string."""
    try:
        if cfg["provider"] == "anthropic":
            import anthropic
            api_key = cfg.get("anthropic_api_key") or None
            client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
            resp = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=20,
                messages=[{
                    "role": "user",
                    "content": (
                        f"User asked: {user_msg[:300]}\n"
                        f"Assistant replied: {assistant_msg[:300]}\n\n"
                        "Write a 3-6 word title for this conversation. "
                        "Reply with ONLY the title, no punctuation."
                    ),
                }],
            )
            return resp.content[0].text.strip()[:80]
    except Exception:
        pass
    return user_msg[:50].rstrip() + ("…" if len(user_msg) > 50 else "")


def _build_chat(cfg: dict, turns: list | None = None):
    if cfg["provider"] == "ollama":
        from chatlas import ChatOllama
        return ChatOllama(
            model=cfg["ollama_model"],
            system_prompt=SYSTEM_PROMPT_EN,
            base_url=cfg["ollama_base_url"],
            turns=turns,
        )
    from chatlas import ChatAnthropic
    kwargs: dict = {"model": cfg["anthropic_model"], "system_prompt": SYSTEM_PROMPT_EN, "turns": turns}
    if cfg.get("anthropic_api_key"):
        kwargs["api_key"] = cfg["anthropic_api_key"]
    return ChatAnthropic(**kwargs)


class CreateChatBody(BaseModel):
    dataset_id: str


class PostMessageBody(BaseModel):
    content: str


@router.get("/chats")
async def list_chats(
    dataset_id: Optional[str] = None,
    user_id: str = Depends(get_current_user_id),
) -> list[dict]:
    summaries = state.list_chats(user_id, dataset_id)
    return [
        {
            "id": s.id,
            "datasetId": s.dataset_id,
            "datasetName": s.dataset_name,
            "title": s.title,
            "messageCount": s.message_count,
            "lastMessageAt": s.last_message_at,
            "createdAt": s.created_at,
            "updatedAt": s.updated_at,
        }
        for s in summaries
    ]


@router.post("/chats")
async def create_chat(
    body: CreateChatBody,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    chat = state.create_chat(user_id, body.dataset_id)
    if not chat:
        raise HTTPException(404, "Dataset not found")
    return {"id": chat.id, "datasetId": chat.dataset_id, "title": chat.title}


@router.delete("/chats/{chat_id}", status_code=204, response_model=None)
async def delete_chat(chat_id: str, user_id: str = Depends(get_current_user_id)) -> None:
    if not state.delete_chat(chat_id, user_id):
        raise HTTPException(404, "Chat not found")


@router.get("/chats/{chat_id}")
async def get_chat(chat_id: str, user_id: str = Depends(get_current_user_id)) -> dict:
    chat = state.get_chat(chat_id, user_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    return {
        "id": chat.id,
        "datasetId": chat.dataset_id,
        "title": chat.title,
        "messages": [state.message_to_dict(m) for m in chat.messages],
    }


@router.post("/chats/{chat_id}/messages")
async def post_message(
    chat_id: str,
    body: PostMessageBody,
    user_id: str = Depends(get_current_user_id),
) -> StreamingResponse:
    chat = state.get_chat(chat_id, user_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    df = state.get_dataset_dataframe(chat.dataset_id, user_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")

    # Persist the user turn + an empty assistant placeholder before streaming.
    state.add_user_message(chat_id, body.content)
    assistant_id = state.create_assistant_placeholder(chat_id)

    # Re-fetch so history reflects what's now in the DB.
    chat = state.get_chat(chat_id, user_id)
    history = _history_turns(chat)

    dataset = state.get_dataset(chat.dataset_id, user_id)
    session = ChatSession(chat_id=chat_id, df=df, data_name=dataset.name if dataset else "")
    cfg = state.get_user_settings(user_id)

    async def stream() -> AsyncIterator[str]:
        set_session(session)
        try:
            chat_obj = _build_chat(cfg, turns=history)
            chat_obj.register_tool(tools.get_data_overview)
            chat_obj.register_tool(tools.run_analysis)
            chat_obj.register_tool(tools.run_sql)
            chat_obj.register_tool(tools.create_chart)
            chat_obj.register_tool(tools.create_interactive_chart)
            chat_obj.register_tool(tools.detect_outliers)
            chat_obj.register_tool(tools.compute_statistics)
            chat_obj.register_tool(tools.create_map)
            chat_obj.register_tool(tools.compute_nps)
            chat_obj.register_tool(tools.add_to_dashboard)
            chat_obj.register_tool(tools.clear_dashboard_tool)
            chat_obj.register_tool(tools.suggest_next_analyses)

            buffer: list[str] = []

            def drain_pending() -> list[str]:
                events: list[str] = []
                # Pair charts with their config (same index in both lists).
                for path in session.pending_charts:
                    title = ""
                    chart_type = ""
                    for cfg in session.pending_chart_configs:
                        if cfg.get("path") == str(path):
                            title = cfg.get("title", "")
                            chart_type = cfg.get("chart_type", "")
                            break
                    rec = state.append_chart(assistant_id, path, title, chart_type)
                    import base64 as _b64
                    data_url = (
                        f"data:image/png;base64,{_b64.b64encode(rec.image_bytes).decode('ascii')}"
                        if rec.image_bytes else ""
                    )
                    events.append(sse("chart", {
                        "id": rec.id,
                        "dataUrl": data_url,
                        "title": title,
                        "chartType": chart_type,
                    }))
                session.pending_charts.clear()
                session.pending_chart_configs.clear()

                for plotly_item in session.pending_plotly_charts:
                    rec = state.append_plotly_chart(assistant_id, plotly_item["spec"], plotly_item["title"])
                    events.append(sse("plotly_chart", {
                        "id": rec.id,
                        "spec": rec.spec_json,
                        "title": rec.title,
                    }))
                session.pending_plotly_charts.clear()

                for snip in session.pending_code_snippets:
                    rec = state.append_snippet(assistant_id, snip.get("type", "analysis"), snip.get("code", ""))
                    events.append(sse("snippet", {
                        "id": rec.id, "type": rec.type, "code": rec.code,
                    }))
                session.pending_code_snippets.clear()

                if session.pending_dashboard_charts:
                    baseline = len(state.get_dashboard(session.chat_id))
                    for i, item in enumerate(session.pending_dashboard_charts):
                        rec = state.append_dashboard_chart(
                            session.chat_id, item["spec"], item["title"], baseline + i
                        )
                        events.append(sse("dashboard_chart", state.dashboard_chart_to_dict(rec)))
                    session.pending_dashboard_charts.clear()

                return events

            for chunk in chat_obj.stream(body.content):
                if chunk:
                    buffer.append(chunk)
                    yield sse("token", {"text": chunk})
                for ev in drain_pending():
                    yield ev
                await asyncio.sleep(0)

            for ev in drain_pending():
                yield ev

            full_text = "".join(buffer)
            state.finalize_assistant_message(assistant_id, full_text)
            yield sse("done", {"messageId": assistant_id})

            # Post-stream: title + suggestions (run in parallel via executor).
            chat_after = state.get_chat(chat_id, user_id)
            real_msgs = [m for m in (chat_after.messages if chat_after else [])
                         if m.role in ("user", "assistant") and m.content.strip()]
            user_msg_text = real_msgs[0].content if real_msgs else body.content
            ds_columns = list(df.columns.astype(str)) if df is not None else []
            loop = asyncio.get_event_loop()

            if len(real_msgs) == 2:
                new_title = await loop.run_in_executor(
                    None, _generate_title, user_msg_text, full_text, cfg
                )
                state.update_chat_title(chat_id, new_title)
                yield sse("title", {"title": new_title})

            if full_text.strip():
                suggestions = await loop.run_in_executor(
                    None, _generate_suggestions, user_msg_text, full_text, ds_columns, cfg
                )
                if suggestions:
                    yield sse("suggestions", {"items": suggestions})

        except Exception as e:
            yield sse("error", {"message": f"{type(e).__name__}: {e}"})
            traceback.print_exc()
            yield sse("done", {"messageId": assistant_id})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/chats/{chat_id}/dashboard")
async def get_dashboard(
    chat_id: str,
    user_id: str = Depends(get_current_user_id),
) -> list[dict]:
    chat = state.get_chat(chat_id, user_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    charts = state.get_dashboard(chat_id)
    return [state.dashboard_chart_to_dict(c) for c in charts]


@router.delete("/chats/{chat_id}/dashboard", status_code=204, response_model=None)
async def delete_dashboard(
    chat_id: str,
    user_id: str = Depends(get_current_user_id),
) -> None:
    chat = state.get_chat(chat_id, user_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    state.clear_dashboard(chat_id)


# Note: /charts/:id used to serve the PNG. It's been retired in favor of
# inlining the data URL directly into SSE events and chat-fetch responses,
# so <img> tags work without an auth round-trip.
