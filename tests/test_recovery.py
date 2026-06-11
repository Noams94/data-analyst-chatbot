"""
Unit tests for api/recovery.py — the parser that finds tool-call JSON the
model emitted as text, and the span-stripper that removes it while keeping
the surrounding prose. Run with:  pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.recovery import extract_tool_call_jsons, strip_spans


# ─── extract_tool_call_jsons ──────────────────────────────────────────────────

def test_no_json_returns_empty():
    assert extract_tool_call_jsons("Just a normal analytical answer.") == []


def test_simple_tool_call():
    text = '{"name": "record_finding", "parameters": {"title": "T", "claim": "C"}}'
    out = extract_tool_call_jsons(text)
    assert len(out) == 1
    obj, start, end = out[0]
    assert obj["name"] == "record_finding"
    assert obj["parameters"]["title"] == "T"
    assert text[start:end] == text


def test_call_embedded_in_prose_spans_match_original():
    text = 'Here are the results.\n{"name": "add_to_dashboard", "parameters": {"x": 1}}\nDone.'
    out = extract_tool_call_jsons(text)
    assert len(out) == 1
    obj, start, end = out[0]
    assert text[start:end].startswith('{"name"')
    assert text[start:end].endswith("}")
    assert obj["parameters"] == {"x": 1}


def test_python_literals_normalized():
    text = '{"name": "record_finding", "parameters": {"chart": None, "flag": True, "other": False}}'
    out = extract_tool_call_jsons(text)
    assert len(out) == 1
    params = out[0][0]["parameters"]
    assert params == {"chart": None, "flag": True, "other": False}


def test_python_literal_normalization_preserves_offsets():
    # None/True/False replacements are same-length, so spans computed on the
    # normalized text must still index correctly into the original.
    text = 'Note: True story.\n{"name": "t", "parameters": {"a": None}} trailing'
    out = extract_tool_call_jsons(text)
    assert len(out) == 1
    _, start, end = out[0]
    assert text[start:end] == '{"name": "t", "parameters": {"a": None}}'


def test_nested_objects_and_braces_in_strings():
    text = '{"name": "t", "parameters": {"spec": {"layout": {"title": "a } b"}}, "n": 2}}'
    out = extract_tool_call_jsons(text)
    assert len(out) == 1
    assert out[0][0]["parameters"]["spec"]["layout"]["title"] == "a } b"


def test_multiple_calls():
    text = (
        'First: {"name": "a", "parameters": {}} and then '
        '{"name": "b", "parameters": {"k": "v"}} done.'
    )
    out = extract_tool_call_jsons(text)
    assert [o[0]["name"] for o in out] == ["a", "b"]


def test_malformed_json_skipped():
    text = '{"name": "broken", "parameters": {unquoted: yes}}'
    assert extract_tool_call_jsons(text) == []


def test_non_tool_objects_with_name_key_still_parse():
    # An object with a string "name" is returned; the caller's whitelist
    # decides whether it maps to an executable tool.
    text = '{"name": "Laptop", "parameters": {"price": 100}}'
    out = extract_tool_call_jsons(text)
    assert len(out) == 1


# ─── strip_spans ─────────────────────────────────────────────────────────────

def test_strip_no_spans_returns_trimmed_text():
    assert strip_spans("  hello  ", []) == "hello"


def test_strip_keeps_surrounding_prose():
    text = 'Revenue is up.\n{"name": "t", "parameters": {}}\nSee the chart.'
    out = extract_tool_call_jsons(text)
    cleaned = strip_spans(text, [(s, e) for _, s, e in out])
    assert "Revenue is up." in cleaned
    assert "See the chart." in cleaned
    assert '"parameters"' not in cleaned


def test_strip_removes_emptied_code_fence():
    text = 'Summary first.\n```json\n{"name": "t", "parameters": {}}\n```\nMore prose.'
    out = extract_tool_call_jsons(text)
    cleaned = strip_spans(text, [(s, e) for _, s, e in out])
    assert "```" not in cleaned
    assert "Summary first." in cleaned
    assert "More prose." in cleaned


def test_strip_everything_yields_empty():
    text = '{"name": "t", "parameters": {}}'
    out = extract_tool_call_jsons(text)
    cleaned = strip_spans(text, [(s, e) for _, s, e in out])
    assert cleaned == ""


def test_strip_collapses_blank_runs():
    text = 'A.\n\n{"name": "t", "parameters": {}}\n\nB.'
    out = extract_tool_call_jsons(text)
    cleaned = strip_spans(text, [(s, e) for _, s, e in out])
    assert "\n\n\n" not in cleaned
    assert cleaned.startswith("A.")
    assert cleaned.endswith("B.")
