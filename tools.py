"""
Analysis tools registered with the chatlas Chat object.
Module-level state is shared within a single Streamlit session.
"""

import io
import json
import traceback
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ─── Module State ─────────────────────────────────────────────────────────────

_df: Optional[pd.DataFrame] = None
_data_name: str = ""
_pending_charts: list[Path] = []      # Charts waiting to be shown in the UI
_charts_dir = Path(__file__).parent / "charts"
_charts_dir.mkdir(exist_ok=True)

# ─── Matplotlib Style ─────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 130,
    "figure.figsize": (10, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.family": "DejaVu Sans",
})
sns.set_palette("husl")

# ─── Public API for Streamlit ─────────────────────────────────────────────────

def set_dataframe(df: pd.DataFrame, name: str = "uploaded_data") -> None:
    """Set the active DataFrame used by all tools."""
    global _df, _data_name
    _df = df
    _data_name = name


def get_dataframe() -> Optional[pd.DataFrame]:
    return _df


def get_data_name() -> str:
    return _data_name


def get_pending_charts() -> list[Path]:
    """Return charts created since the last call, then clear the queue."""
    charts = list(_pending_charts)
    _pending_charts.clear()
    return charts


# ─── Registered Tools ─────────────────────────────────────────────────────────

def get_data_overview() -> str:
    """
    Get a comprehensive overview of the loaded dataset.
    Returns: column types, missing values, numeric stats, and sample rows.
    Always call this first before any analysis.
    """
    if _df is None:
        return "⚠️ No dataset loaded. Please upload a file."

    buf = io.StringIO()
    buf.write(f"File: {_data_name}\n")
    buf.write(f"Shape: {_df.shape[0]:,} rows × {_df.shape[1]} columns\n\n")

    # Column info table
    rows = []
    for col in _df.columns:
        nulls = _df[col].isna().sum()
        rows.append({
            "Column": col,
            "Type": str(_df[col].dtype),
            "Non-null": f"{len(_df) - nulls:,}",
            "Null%": f"{nulls / len(_df) * 100:.1f}%",
            "Unique": f"{_df[col].nunique():,}",
        })
    buf.write("Column Info:\n")
    buf.write(pd.DataFrame(rows).to_string(index=False))
    buf.write("\n\n")

    # Numeric stats
    num_cols = _df.select_dtypes(include="number").columns
    if len(num_cols):
        buf.write("Numeric Summary:\n")
        buf.write(_df[num_cols].describe().round(2).to_string())
        buf.write("\n\n")

    # Top values for categoricals
    cat_cols = _df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols):
        buf.write("Top Values per Categorical Column:\n")
        for col in cat_cols[:6]:
            top = _df[col].value_counts().head(5)
            buf.write(f"  {col}: {dict(top)}\n")
        buf.write("\n")

    buf.write("Sample (first 5 rows):\n")
    buf.write(_df.head(5).to_string(index=False))
    return buf.getvalue()


def run_analysis(pandas_code: str) -> str:
    """
    Run Python/pandas code on the DataFrame (variable name: `df`) and return the result.
    Use for aggregations, groupby, correlations, filtering, pivot tables, etc.

    Examples:
        "df.groupby('category')['revenue'].sum().sort_values(ascending=False).head(10)"
        "df[df['age'] > 30].describe()"
        "df.corr(numeric_only=True).round(2)"
        "pd.crosstab(df['region'], df['category'])"
    """
    if _df is None:
        return "⚠️ No dataset loaded."

    local_ns = {"df": _df.copy(), "pd": pd, "np": np, "json": json}
    try:
        lines = [l for l in pandas_code.strip().splitlines() if l.strip()]
        if not lines:
            return "No code provided."
        if len(lines) > 1:
            exec("\n".join(lines[:-1]), local_ns)
        result = eval(lines[-1], local_ns)
        if isinstance(result, pd.DataFrame):
            return result.to_string(max_rows=50, max_cols=20)
        elif isinstance(result, pd.Series):
            return result.to_string(max_rows=50)
        elif result is None:
            return "(No output — code executed successfully)"
        return str(result)
    except Exception:
        return f"Error:\n{traceback.format_exc()}"


def create_chart(
    chart_type: str,
    x_column: str,
    y_column: str = "",
    title: str = "",
    color_column: str = "",
    aggregation: str = "sum",
    top_n: int = 0,
    bins: int = 20,
    sort: bool = True,
) -> str:
    """
    Create a chart from the dataset and display it in the UI.

    chart_type:
        "bar"     – bar chart (x=category, y=numeric, grouped & aggregated)
        "barh"    – horizontal bar chart (better for many/long categories)
        "line"    – line chart (x=ordered/date, y=numeric)
        "scatter" – scatter plot (x=numeric, y=numeric, optional color_column)
        "hist"    – histogram for one column (set x_column)
        "box"     – box plot distribution (x=category optional, y=numeric)
        "pie"     – pie chart
        "heatmap" – correlation heatmap (no x/y required)
        "count"   – count of values in x_column

    aggregation: "sum" | "mean" | "count" | "median" | "max" | "min"
    top_n: if > 0, keep only top N categories
    sort: sort bars by value (default True)
    """
    if _df is None:
        return "⚠️ No dataset loaded."

    df = _df.copy()
    fig, ax = plt.subplots()
    t = chart_type.lower().strip()

    try:
        if t == "heatmap":
            num_df = df.select_dtypes(include="number")
            corr = num_df.corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                        center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
            ax.set_title(title or "Correlation Heatmap")

        elif t == "hist":
            df[x_column].dropna().plot.hist(bins=bins, ax=ax,
                                             edgecolor="white", linewidth=0.5)
            ax.set_xlabel(x_column)
            ax.set_ylabel("Frequency")
            ax.set_title(title or f"Distribution of {x_column}")

        elif t in ("bar", "barh", "line"):
            agg = aggregation or "sum"
            if y_column:
                grp = df.groupby(x_column)[y_column].agg(agg).reset_index()
                grp.columns = [x_column, y_column]
                if sort:
                    grp = grp.sort_values(y_column, ascending=(t == "barh"))
                if top_n > 0:
                    grp = grp.tail(top_n) if t == "barh" else grp.head(top_n)
                xv, yv = grp[x_column], grp[y_column]
                ylabel = f"{agg}({y_column})"
            else:
                vc = df[x_column].value_counts()
                if sort:
                    vc = vc.sort_values(ascending=(t == "barh"))
                if top_n > 0:
                    vc = vc.tail(top_n) if t == "barh" else vc.head(top_n)
                xv, yv = vc.index, vc.values
                ylabel = "Count"

            if t == "line":
                ax.plot(range(len(xv)), yv, marker="o", linewidth=2, markersize=5)
                ax.fill_between(range(len(xv)), yv, alpha=0.1)
                ax.set_xticks(range(len(xv)))
                ax.set_xticklabels(xv, rotation=45, ha="right")
                ax.set_ylabel(ylabel)
            elif t == "barh":
                bars = ax.barh(range(len(xv)), yv)
                ax.set_yticks(range(len(xv)))
                ax.set_yticklabels(xv, fontsize=9)
                ax.set_xlabel(ylabel)
            else:
                ax.bar(range(len(xv)), yv)
                ax.set_xticks(range(len(xv)))
                ax.set_xticklabels(xv, rotation=45, ha="right", fontsize=9)
                ax.set_ylabel(ylabel)

            if t != "barh":
                ax.set_xlabel(x_column)
            ax.set_title(title or f"{ylabel} by {x_column}")

        elif t == "scatter":
            kw = {"alpha": 0.55, "s": 35, "edgecolors": "none"}
            if color_column and color_column in df.columns:
                for val, grp in df.groupby(color_column):
                    ax.scatter(grp[x_column], grp[y_column], label=str(val), **kw)
                ax.legend(title=color_column, bbox_to_anchor=(1.02, 1), loc="upper left",
                          fontsize=8, title_fontsize=8)
            else:
                ax.scatter(df[x_column], df[y_column], **kw)
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_title(title or f"{y_column} vs {x_column}")

        elif t == "box":
            if x_column and x_column in df.columns:
                groups = df[x_column].unique()
                data = [df[df[x_column] == g][y_column].dropna() for g in groups]
                bp = ax.boxplot(data, labels=groups, patch_artist=True)
                colors = sns.color_palette("husl", len(groups))
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color + (0.5,))
            else:
                ax.boxplot(df[y_column].dropna(), patch_artist=True)
            plt.xticks(rotation=40, ha="right", fontsize=9)
            ax.set_ylabel(y_column)
            ax.set_title(title or f"Distribution of {y_column}")

        elif t in ("pie", "count"):
            if t == "pie":
                vals = (df.groupby(x_column)[y_column].agg(aggregation or "sum")
                        if y_column else df[x_column].value_counts())
                if top_n > 0:
                    vals = vals.head(top_n)
                vals.plot.pie(ax=ax, autopct="%1.1f%%", startangle=90,
                              pctdistance=0.8, labeldistance=1.1)
                ax.set_ylabel("")
            else:
                vc = df[x_column].value_counts()
                if top_n > 0:
                    vc = vc.head(top_n)
                vc.plot.bar(ax=ax)
                plt.xticks(rotation=45, ha="right", fontsize=9)
                ax.set_ylabel("Count")
            ax.set_title(title or f"Distribution of {x_column}")

        else:
            return (f"Unknown chart_type: '{chart_type}'. "
                    "Valid: bar, barh, line, scatter, hist, box, pie, heatmap, count")

        plt.tight_layout()

        # Save to file
        safe = "".join(c for c in (title or f"{t}_{x_column}") if c.isalnum() or c in "_-")[:48]
        path = _charts_dir / f"{safe}.png"
        idx = 1
        while path.exists():
            path = _charts_dir / f"{safe}_{idx}.png"
            idx += 1

        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        _pending_charts.append(path)
        return f"✅ Chart created: '{title or t}' — displayed below."

    except Exception:
        plt.close(fig)
        return f"Chart error:\n{traceback.format_exc()}"


def suggest_next_analyses() -> str:
    """
    Examine the dataset and suggest meaningful follow-up analyses.
    Call to get ideas for what to explore next.
    """
    if _df is None:
        return "⚠️ No dataset loaded."

    suggestions = []
    num_cols = list(_df.select_dtypes(include="number").columns)
    cat_cols = list(_df.select_dtypes(include=["object", "category"]).columns)
    date_cols = [c for c in _df.columns
                 if "date" in c.lower() or "time" in c.lower()
                 or pd.api.types.is_datetime64_any_dtype(_df[c])]

    if len(num_cols) >= 2:
        suggestions.append(f"📈 Correlation matrix of numeric columns: {num_cols[:4]}")
    if cat_cols and num_cols:
        suggestions.append(f"📊 Compare {num_cols[0]} across categories of '{cat_cols[0]}'")
    if date_cols and num_cols:
        suggestions.append(f"📅 Time series of {num_cols[0]} over '{date_cols[0]}'")
    if len(cat_cols) > 1:
        suggestions.append(f"🔍 Cross-tab: '{cat_cols[0]}' vs '{cat_cols[1]}'")
    if num_cols:
        suggestions.append(f"📉 Distribution check for {num_cols[:3]}")

    missing = _df.isna().sum()
    if missing.sum() > 0:
        top = missing[missing > 0].sort_values(ascending=False).head(3)
        suggestions.append(f"🔎 Missing value analysis: {dict(top)}")

    if num_cols:
        col = num_cols[0]
        outliers = ((_df[col] - _df[col].mean()).abs() > 3 * _df[col].std()).sum()
        if outliers > 0:
            suggestions.append(f"⚠️ Potential outliers in '{col}': {outliers} rows")

    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(suggestions))
