"""Per-request chat session — replaces the module-global state in the legacy tools.py.

Each FastAPI request that runs chatlas tools must `set_session()` first so that the
tool functions can read the right DataFrame and append to the right pending queues.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class ChatSession:
    chat_id: str
    df: pd.DataFrame
    data_name: str = ""
    pending_charts: list[Path] = field(default_factory=list)
    pending_chart_configs: list[dict] = field(default_factory=list)
    pending_code_snippets: list[dict] = field(default_factory=list)


_session: ContextVar[Optional[ChatSession]] = ContextVar("session", default=None)


def set_session(session: ChatSession) -> None:
    _session.set(session)


def current() -> ChatSession:
    sess = _session.get()
    if sess is None:
        raise RuntimeError("No ChatSession bound to this request")
    return sess


def current_or_none() -> Optional[ChatSession]:
    return _session.get()
