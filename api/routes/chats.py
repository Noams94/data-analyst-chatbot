"""Chat creation + streaming message route (SSE)."""

from __future__ import annotations

import asyncio
import os
import traceback
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from api import state, tools
from api.chat_session import ChatSession, set_session
from api.prompts import SYSTEM_PROMPT_EN
from api.sse import sse

router = APIRouter(tags=["chats"])

# Provider selection. Set PROVIDER=anthropic|ollama in api/.env.
# Default: prefer anthropic if a key is present, else ollama.
PROVIDER = os.getenv("PROVIDER", "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "ollama")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


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


def _build_chat(turns: list | None = None):
    if PROVIDER == "ollama":
        from chatlas import ChatOllama
        return ChatOllama(
            model=OLLAMA_MODEL,
            system_prompt=SYSTEM_PROMPT_EN,
            base_url=OLLAMA_URL,
            turns=turns,
        )
    from chatlas import ChatAnthropic
    return ChatAnthropic(model=ANTHROPIC_MODEL, system_prompt=SYSTEM_PROMPT_EN, turns=turns)


class CreateChatBody(BaseModel):
    dataset_id: str


class PostMessageBody(BaseModel):
    content: str


@router.post("/chats")
async def create_chat(body: CreateChatBody) -> dict:
    chat = state.create_chat(body.dataset_id)
    if not chat:
        raise HTTPException(404, "Dataset not found")
    return {"id": chat.id, "datasetId": chat.dataset_id, "title": chat.title}


@router.get("/chats/{chat_id}")
async def get_chat(chat_id: str) -> dict:
    chat = state.get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    return {
        "id": chat.id,
        "datasetId": chat.dataset_id,
        "title": chat.title,
        "messages": [state.message_to_dict(m) for m in chat.messages],
    }


@router.post("/chats/{chat_id}/messages")
async def post_message(chat_id: str, body: PostMessageBody) -> StreamingResponse:
    chat = state.get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    df = state.get_dataset_dataframe(chat.dataset_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")

    # Persist the user turn + an empty assistant placeholder before streaming.
    state.add_user_message(chat_id, body.content)
    assistant_id = state.create_assistant_placeholder(chat_id)

    # Re-fetch so history reflects what's now in the DB.
    chat = state.get_chat(chat_id)
    history = _history_turns(chat)

    dataset = state.get_dataset(chat.dataset_id)
    session = ChatSession(chat_id=chat_id, df=df, data_name=dataset.name if dataset else "")

    async def stream() -> AsyncIterator[str]:
        set_session(session)
        try:
            chat_obj = _build_chat(turns=history)
            chat_obj.register_tool(tools.get_data_overview)
            chat_obj.register_tool(tools.run_analysis)
            chat_obj.register_tool(tools.run_sql)
            chat_obj.register_tool(tools.create_chart)
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
                    events.append(sse("chart", {
                        "id": rec.id,
                        "url": f"/charts/{rec.id}",
                        "title": title,
                        "chartType": chart_type,
                    }))
                session.pending_charts.clear()
                session.pending_chart_configs.clear()

                for snip in session.pending_code_snippets:
                    rec = state.append_snippet(assistant_id, snip.get("type", "analysis"), snip.get("code", ""))
                    events.append(sse("snippet", {
                        "id": rec.id, "type": rec.type, "code": rec.code,
                    }))
                session.pending_code_snippets.clear()
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

            state.finalize_assistant_message(assistant_id, "".join(buffer))
            yield sse("done", {"messageId": assistant_id})

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


@router.get("/charts/{chart_id}")
async def get_chart(chart_id: str) -> FileResponse:
    rec = state.get_chart(chart_id)
    if not rec or not rec.path.exists():
        raise HTTPException(404, "Chart not found")
    return FileResponse(rec.path, media_type="image/png")
