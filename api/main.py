"""FastAPI entry point for the data-analyst chatbot backend."""

from pathlib import Path

from dotenv import load_dotenv

# Load api/.env (if present) before importing modules that read env vars.
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from api.db import init_db  # noqa: E402
from api.routes import chats, datasets, messages, reports  # noqa: E402

init_db()

app = FastAPI(title="data-analyst-chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # tightened in production via env
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(chats.router)
app.include_router(messages.router)
app.include_router(reports.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
