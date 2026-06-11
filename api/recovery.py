"""Recovery of tool calls the model emitted as text instead of invoking.

Pure text-processing helpers — no FastAPI/DB imports — so they can be
unit-tested in isolation. Actual tool execution lives in routes/chats.py.
"""

from __future__ import annotations

import json
import re

# Python-literal → JSON-literal replacements. Each pair has equal length, so
# character offsets into the normalized text remain valid in the original.
_PY_LITERALS = (
    (r"\bNone\b", "null"),
    (r"\bTrue\b", "true"),
    (r"\bFalse\b", "false"),
)


def extract_tool_call_jsons(text: str) -> list:
    """Find every {"name": "<tool>", "parameters": {...}} object embedded in
    the text, even when wrapped in prose or markdown. Returns a list of
    (obj, start, end) tuples; brace-matching handles nested objects."""
    for pat, repl in _PY_LITERALS:
        text = re.sub(pat, repl, text)

    out: list = []
    i = 0
    while True:
        m = re.search(r'\{\s*"name"\s*:', text[i:])
        if not m:
            break
        start = i + m.start()
        # Walk braces (respect strings) to find the matching close.
        depth = 0
        in_str = False
        esc = False
        end = -1
        for j in range(start, len(text)):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break
        if end < 0:
            break
        chunk = text[start:end]
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict) and isinstance(obj.get("name"), str):
                out.append((obj, start, end))
        except Exception:
            pass
        i = end
    return out


def strip_spans(text: str, spans: list) -> str:
    """Remove (start, end) spans from text, then tidy what's left: drop code
    fences emptied by the removal and collapse runs of blank lines."""
    if not spans:
        return text.strip()
    parts: list[str] = []
    pos = 0
    for start, end in sorted(spans):
        parts.append(text[pos:start])
        pos = end
    parts.append(text[pos:])
    cleaned = "".join(parts)
    cleaned = re.sub(r"```[\w-]*\s*```", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
