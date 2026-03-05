"""
Unit tests for tools.py — run with:  pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the parent package importable
sys.path.insert(0, str(Path(__file__).parent.parent))
import tools


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def sample_df():
    """Load a small, deterministic DataFrame before each test."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "product":  ["A", "B", "C", "A", "B"] * 20,
        "sales":    rng.integers(100, 1000, 100).astype(float),
        "month":    (["January", "February", "March", "April", "May"] * 20),
        "quantity": rng.integers(1, 50, 100).astype(float),
    })
    tools.set_dataframe(df, name="test_data.csv")
    return df


# ─── set_dataframe / get_dataframe ────────────────────────────────────────────

def test_set_and_get_dataframe(sample_df):
    df = tools.get_dataframe()
    assert df is not None
    assert list(df.columns) == ["product", "sales", "month", "quantity"]


def test_get_data_name():
    assert tools.get_data_name() == "test_data.csv"


# ─── get_data_overview ────────────────────────────────────────────────────────

def test_get_data_overview_returns_string():
    result = tools.get_data_overview()
    assert isinstance(result, str)
    assert len(result) > 50


def test_get_data_overview_contains_shape():
    result = tools.get_data_overview()
    assert "100" in result   # 100 rows


# ─── run_analysis ─────────────────────────────────────────────────────────────

def test_run_analysis_groupby():
    code = "df.groupby('product')['sales'].sum().to_string()"
    result = tools.run_analysis(code)
    assert isinstance(result, str)
    assert "A" in result and "B" in result and "C" in result


def test_run_analysis_error_returns_message():
    result = tools.run_analysis("this is not valid python!!!")
    assert "error" in result.lower() or "Error" in result


def test_run_analysis_no_df(monkeypatch):
    monkeypatch.setattr(tools, "_df", None)
    result = tools.run_analysis("df.head()")
    assert "no data" in result.lower() or "נתונים" in result or "error" in result.lower()


# ─── _fix_rtl ─────────────────────────────────────────────────────────────────

def test_fix_rtl_english_unchanged():
    text = "Hello World"
    assert tools._fix_rtl(text) == text


def test_fix_rtl_empty_string():
    assert tools._fix_rtl("") == ""


def test_fix_rtl_hebrew_not_empty():
    # Just verify it returns a non-empty string for Hebrew input
    result = tools._fix_rtl("שלום עולם")
    assert isinstance(result, str) and len(result) > 0


# ─── _sort_chronological ──────────────────────────────────────────────────────

def test_sort_chronological_english_months():
    df = pd.DataFrame({
        "month": ["March", "January", "February"],
        "val":   [3, 1, 2],
    })
    result = tools._sort_chronological(df, "month")
    assert result is not None
    assert list(result["month"]) == ["January", "February", "March"]


def test_sort_chronological_no_months():
    df = pd.DataFrame({"cat": ["X", "Y", "Z"], "val": [1, 2, 3]})
    result = tools._sort_chronological(df, "cat")
    assert result is None


def test_sort_chronological_hebrew_months():
    df = pd.DataFrame({
        "month": ["מרץ", "ינואר", "פברואר"],
        "val":   [3, 1, 2],
    })
    result = tools._sort_chronological(df, "month")
    assert result is not None
    assert list(result["month"]) == ["ינואר", "פברואר", "מרץ"]


# ─── create_chart ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("chart_type", ["bar", "line", "hist", "pie", "count"])
def test_create_chart_returns_path(tmp_path, monkeypatch, chart_type):
    monkeypatch.setattr(tools, "_charts_dir", tmp_path)
    tools._pending_charts.clear()

    result = tools.create_chart(
        chart_type=chart_type,
        x_column="product" if chart_type in ("bar", "pie", "count") else "sales",
        y_column="sales" if chart_type == "bar" else None,
        title=f"Test {chart_type}",
    )
    assert isinstance(result, str)
    # A chart file should have been created
    charts = list(tmp_path.glob("*.png"))
    assert len(charts) >= 1, f"No PNG created for chart_type={chart_type}"


def test_create_chart_scatter(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "_charts_dir", tmp_path)
    tools._pending_charts.clear()

    result = tools.create_chart(
        chart_type="scatter",
        x_column="sales",
        y_column="quantity",
        title="Scatter test",
    )
    assert isinstance(result, str)
    assert len(list(tmp_path.glob("*.png"))) >= 1


def test_create_chart_no_data(monkeypatch):
    monkeypatch.setattr(tools, "_df", None)
    result = tools.create_chart(chart_type="bar", x_column="product")
    assert "no data" in result.lower() or "נתונים" in result or "error" in result.lower()


# ─── suggest_next_analyses ────────────────────────────────────────────────────

def test_suggest_next_analyses_returns_string():
    result = tools.suggest_next_analyses()
    assert isinstance(result, str) and len(result) > 10
