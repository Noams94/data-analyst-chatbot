"""Server-Sent Events helpers."""

from __future__ import annotations

import json
from typing import Any


def sse(event: str, data: dict[str, Any]) -> str:
    """Format a single SSE frame.

    Both `event:` and `data:` lines are required. Each frame ends with a blank line.
    """
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"
