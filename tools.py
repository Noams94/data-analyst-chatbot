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
_pending_charts: list[Path] = []        # Charts waiting to be shown in the UI
_pending_chart_configs: list[dict] = [] # Config dicts (type, x, y, …) per chart
_pending_code_snippets: list[dict] = []  # {"type": "analysis"|"chart", "code": str}
_charts_dir = Path(__file__).parent / "charts"
_charts_dir.mkdir(exist_ok=True)

# ─── AI Dashboard State ──────────────────────────────────────────────────────
_ai_dashboard_charts: list[dict] = []

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

# ─── Chart Style State ────────────────────────────────────────────────────────

_chart_style: dict = {
    "seaborn_style": "whitegrid",
    "palette": "husl",
    "dpi": 130,
    "figsize": (10, 5),
}

# Month name → sort index (English + Hebrew)
_MONTH_ORDER: dict = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני",
    "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר",
])}

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


def set_chart_style(
    seaborn_style: str = "whitegrid",
    palette: str = "husl",
    dpi: int = 130,
    figsize: tuple = (10, 5),
) -> None:
    """Update chart style preferences (called from the Streamlit UI)."""
    global _chart_style
    _chart_style = {
        "seaborn_style": seaborn_style,
        "palette": palette,
        "dpi": dpi,
        "figsize": figsize,
    }
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["figure.figsize"] = list(figsize)
    sns.set_palette(palette)
    sns.set_style(seaborn_style)


def _fix_rtl(text: str) -> str:
    """Reorder Hebrew/Arabic RTL text for matplotlib's left-to-right renderer."""
    if not text:
        return text
    try:
        from bidi.algorithm import get_display
        return get_display(text)
    except ImportError:
        return text


def _sort_chronological(grp_df: pd.DataFrame, x_col: str):
    """
    If x_col contains month names, return the df sorted chronologically.
    Returns None when no month names are detected.
    """
    sample = [str(v).lower() for v in grp_df[x_col].head(12)]
    if not any(v in _MONTH_ORDER for v in sample):
        return None
    order = grp_df[x_col].apply(lambda v: _MONTH_ORDER.get(str(v).lower(), 99))
    return grp_df.iloc[order.argsort().values].reset_index(drop=True)


def get_pending_charts() -> list[Path]:
    """Return charts created since the last call, then clear the queue."""
    charts = list(_pending_charts)
    _pending_charts.clear()
    return charts


def get_pending_chart_configs() -> list[dict]:
    """Return chart configs created since the last call, then clear the queue."""
    configs = list(_pending_chart_configs)
    _pending_chart_configs.clear()
    return configs


def get_pending_code_snippets() -> list[dict]:
    """Return code snippets since the last call, then clear the queue.
    Each item: {"type": "analysis"|"chart", "code": str}
    """
    snippets = list(_pending_code_snippets)
    _pending_code_snippets.clear()
    return snippets


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

    _pending_code_snippets.append({"type": "analysis", "code": pandas_code.strip()})
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
    sns.set_style(_chart_style["seaborn_style"])
    sns.set_palette(_chart_style["palette"])
    fig, ax = plt.subplots(
        figsize=_chart_style["figsize"],
        dpi=_chart_style["dpi"],
    )
    t = chart_type.lower().strip()

    try:
        if t == "heatmap":
            num_df = df.select_dtypes(include="number")
            corr = num_df.corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                        center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
            ax.set_title(_fix_rtl(title) if title else "Correlation Heatmap")

        elif t == "hist":
            df[x_column].dropna().plot.hist(bins=bins, ax=ax,
                                             edgecolor="white", linewidth=0.5)
            ax.set_xlabel(_fix_rtl(x_column))
            ax.set_ylabel("Frequency")
            ax.set_title(_fix_rtl(title) if title else _fix_rtl(f"Distribution of {x_column}"))

        elif t in ("bar", "barh", "line"):
            agg = aggregation or "sum"
            if y_column:
                grp = df.groupby(x_column)[y_column].agg(agg).reset_index()
                grp.columns = [x_column, y_column]
                # Prefer chronological order when x contains month names
                chr_grp = _sort_chronological(grp, x_column)
                if chr_grp is not None:
                    grp = chr_grp
                elif sort:
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

            xv_rtl = [_fix_rtl(str(v)) for v in xv]
            if t == "line":
                ax.plot(range(len(xv)), yv, marker="o", linewidth=2, markersize=5)
                ax.fill_between(range(len(xv)), yv, alpha=0.1)
                ax.set_xticks(range(len(xv)))
                ax.set_xticklabels(xv_rtl, rotation=45, ha="right")
                ax.set_ylabel(_fix_rtl(ylabel))
            elif t == "barh":
                ax.barh(range(len(xv)), yv)
                ax.set_yticks(range(len(xv)))
                ax.set_yticklabels(xv_rtl, fontsize=9)
                ax.set_xlabel(_fix_rtl(ylabel))
            else:
                ax.bar(range(len(xv)), yv)
                ax.set_xticks(range(len(xv)))
                ax.set_xticklabels(xv_rtl, rotation=45, ha="right", fontsize=9)
                ax.set_ylabel(_fix_rtl(ylabel))

            if t != "barh":
                ax.set_xlabel(_fix_rtl(x_column))
            ax.set_title(_fix_rtl(title or f"{ylabel} by {x_column}"))

        elif t == "scatter":
            kw = {"alpha": 0.55, "s": 35, "edgecolors": "none"}
            if color_column and color_column in df.columns:
                for val, grp in df.groupby(color_column):
                    ax.scatter(grp[x_column], grp[y_column], label=_fix_rtl(str(val)), **kw)
                ax.legend(title=_fix_rtl(color_column), bbox_to_anchor=(1.02, 1),
                          loc="upper left", fontsize=8, title_fontsize=8)
            else:
                ax.scatter(df[x_column], df[y_column], **kw)
            ax.set_xlabel(_fix_rtl(x_column))
            ax.set_ylabel(_fix_rtl(y_column))
            ax.set_title(_fix_rtl(title) if title else _fix_rtl(f"{y_column} vs {x_column}"))

        elif t == "box":
            if x_column and x_column in df.columns:
                groups = df[x_column].unique()
                data = [df[df[x_column] == g][y_column].dropna() for g in groups]
                bp = ax.boxplot(data, labels=[_fix_rtl(str(g)) for g in groups],
                                patch_artist=True)
                colors = sns.color_palette(_chart_style["palette"], len(groups))
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color + (0.5,))
            else:
                ax.boxplot(df[y_column].dropna(), patch_artist=True)
            plt.xticks(rotation=40, ha="right", fontsize=9)
            ax.set_ylabel(_fix_rtl(y_column))
            ax.set_title(_fix_rtl(title) if title else _fix_rtl(f"Distribution of {y_column}"))

        elif t in ("pie", "count"):
            if t == "pie":
                vals = (df.groupby(x_column)[y_column].agg(aggregation or "sum")
                        if y_column else df[x_column].value_counts())
                if top_n > 0:
                    vals = vals.head(top_n)
                vals.index = [_fix_rtl(str(v)) for v in vals.index]
                vals.plot.pie(ax=ax, autopct="%1.1f%%", startangle=90,
                              pctdistance=0.8, labeldistance=1.1)
                ax.set_ylabel("")
            else:
                vc = df[x_column].value_counts()
                if top_n > 0:
                    vc = vc.head(top_n)
                vc.index = [_fix_rtl(str(v)) for v in vc.index]
                vc.plot.bar(ax=ax)
                plt.xticks(rotation=45, ha="right", fontsize=9)
                ax.set_ylabel("Count")
            ax.set_title(_fix_rtl(title) if title else _fix_rtl(f"Distribution of {x_column}"))

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

        # Track config (for "Add to Dashboard") and code (for code viewer)
        _pending_chart_configs.append({
            "type": "image",
            "path": str(path),
            "title": title or f"{t} – {x_column}",
        })
        _code = (
            f"create_chart(\n"
            f"    chart_type={chart_type!r},\n"
            f"    x_column={x_column!r},\n"
            + (f"    y_column={y_column!r},\n" if y_column else "")
            + (f"    title={title!r},\n" if title else "")
            + (f"    color_column={color_column!r},\n" if color_column else "")
            + (f"    aggregation={aggregation!r},\n" if aggregation != "sum" else "")
            + (f"    top_n={top_n},\n" if top_n else "")
            + ")"
        )
        _pending_code_snippets.append({"type": "chart", "code": _code})

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


# ═══════════════════════════════════════════════════════════════════════════════
# AI Dashboard Builder — tools
# ═══════════════════════════════════════════════════════════════════════════════

# Chart type normalisation: accept short names and map to full bilingual strings
_CHART_TYPE_MAP: dict[str, str] = {
    "bar":       "עמודות / Bar",
    "Bar":       "עמודות / Bar",
    "עמודות":    "עמודות / Bar",
    "line":      "קו / Line",
    "Line":      "קו / Line",
    "קו":        "קו / Line",
    "area":      "שטח / Area",
    "Area":      "שטח / Area",
    "שטח":       "שטח / Area",
    "pie":       "עוגה / Pie",
    "Pie":       "עוגה / Pie",
    "עוגה":      "עוגה / Pie",
    "histogram": "היסטוגרמה / Histogram",
    "Histogram": "היסטוגרמה / Histogram",
    "hist":      "היסטוגרמה / Histogram",
    "היסטוגרמה": "היסטוגרמה / Histogram",
    "scatter":   "פיזור / Scatter",
    "Scatter":   "פיזור / Scatter",
    "פיזור":     "פיזור / Scatter",
    "box":       "Box Plot",
    "box plot":  "Box Plot",
    "heatmap":   "Heatmap",
}

_VALID_CHART_TYPES = {
    "עמודות / Bar", "קו / Line", "שטח / Area", "עוגה / Pie",
    "היסטוגרמה / Histogram", "פיזור / Scatter", "Box Plot", "Heatmap",
}

# Palette normalisation: accept English-only names
_PALETTE_MAP: dict[str, str] = {
    "Blue":   "כחול / Blue",
    "blue":   "כחול / Blue",
    "Purple": "סגול / Purple",
    "purple": "סגול / Purple",
    "Green":  "ירוק / Green",
    "green":  "ירוק / Green",
    "Orange": "כתום / Orange",
    "orange": "כתום / Orange",
    "Pink":   "ורוד / Pink",
    "pink":   "ורוד / Pink",
    "Teal":   "ציאן / Teal",
    "teal":   "ציאן / Teal",
}


def _normalise_chart_type(t: str) -> str:
    """Map common abbreviations to the full bilingual chart-type string."""
    if t in _VALID_CHART_TYPES:
        return t
    return _CHART_TYPE_MAP.get(t, _CHART_TYPE_MAP.get(t.lower().strip(), t))


def _normalise_palette(p: str) -> str:
    """Map English-only palette name to the internal bilingual key."""
    return _PALETTE_MAP.get(p, p)


def _validate_chart_config(cfg: dict) -> Optional[str]:
    """Return an error string if cfg references invalid columns, else None."""
    if _df is None:
        return "No dataset loaded"
    valid_cols = set(_df.columns.tolist())
    x = cfg.get("x", "")
    y = cfg.get("y", "—")
    color = cfg.get("color")
    if x and x not in valid_cols:
        return f"x column '{x}' not found in dataset"
    if y and y != "—" and y not in valid_cols:
        return f"y column '{y}' not found in dataset"
    if color and color not in valid_cols:
        return f"color column '{color}' not found in dataset"
    return None


def _apply_chart_defaults(cfg: dict) -> dict:
    """Fill in missing optional fields with sensible defaults."""
    cfg.setdefault("palette", "כחול / Blue")
    cfg.setdefault("sample_size", 500)
    cfg.setdefault("color", None)
    cfg.setdefault("y", "—")
    cfg.setdefault("corr_method", "pearson")
    if "type" in cfg:
        cfg["type"] = _normalise_chart_type(cfg["type"])
    if "palette" in cfg:
        cfg["palette"] = _normalise_palette(cfg["palette"])
    return cfg


def get_ai_dashboard_charts() -> list[dict]:
    """Return the current AI dashboard chart configs."""
    return list(_ai_dashboard_charts)


def clear_ai_dashboard_charts() -> None:
    """Clear all AI dashboard charts."""
    _ai_dashboard_charts.clear()


def set_dashboard_charts(charts: list[dict]) -> str:
    """
    Replace the entire AI dashboard with the given list of chart configurations.
    Use this for initial dashboard creation or full rebuilds.

    charts: list of dicts, each with keys:
        type (str): Chart type — one of "עמודות / Bar", "קו / Line", "שטח / Area",
                    "עוגה / Pie", "היסטוגרמה / Histogram", "פיזור / Scatter",
                    "Box Plot", "Heatmap". Short names like "bar", "line" are also accepted.
        x (str): Column name for X-axis
        y (str): Column name for Y-axis (numeric), or "—" if not needed
        title (str): Descriptive chart title in the user's language
        color (str, optional): Column name for color grouping
        palette (str, optional): Color palette — "כחול / Blue", "סגול / Purple",
                 "ירוק / Green", "כתום / Orange", "ורוד / Pink", "ציאן / Teal"
        corr_method (str, optional): For Heatmap only — "pearson", "spearman", or "kendall"
        sample_size (int, optional): Number of rows to plot (default 500)

    Returns a confirmation message.
    """
    global _ai_dashboard_charts
    if _df is None:
        return "⚠️ No dataset loaded."

    errors: list[str] = []
    for i, cfg in enumerate(charts):
        cfg = _apply_chart_defaults(cfg)
        err = _validate_chart_config(cfg)
        if err:
            errors.append(f"Chart {i + 1}: {err}")
        charts[i] = cfg

    if errors:
        return "⚠️ Validation errors:\n" + "\n".join(errors)

    _ai_dashboard_charts = list(charts)
    return f"✅ Dashboard created with {len(charts)} charts."


def add_dashboard_chart(chart: dict) -> str:
    """
    Add a single chart to the existing AI dashboard.

    chart: dict with keys type, x, y, title, and optionally color, palette, etc.
           Same format as set_dashboard_charts.

    Returns confirmation with the chart's position index.
    """
    if _df is None:
        return "⚠️ No dataset loaded."
    chart = _apply_chart_defaults(chart)
    err = _validate_chart_config(chart)
    if err:
        return f"⚠️ {err}"
    _ai_dashboard_charts.append(chart)
    return (
        f"✅ Chart added at position {len(_ai_dashboard_charts)}. "
        f"Dashboard now has {len(_ai_dashboard_charts)} charts."
    )


def update_dashboard_chart(index: int, updates: dict) -> str:
    """
    Update a specific chart in the AI dashboard by its 0-based index.
    Only the provided fields are changed; others remain as-is.

    index: 0-based position of the chart to update
    updates: dict of fields to change, e.g. {"type": "קו / Line", "title": "New Title"}

    Returns confirmation or error.
    """
    if index < 0 or index >= len(_ai_dashboard_charts):
        return (
            f"⚠️ Invalid chart index {index}. "
            f"Dashboard has {len(_ai_dashboard_charts)} charts (indices 0–{len(_ai_dashboard_charts) - 1})."
        )
    if "type" in updates:
        updates["type"] = _normalise_chart_type(updates["type"])
    # Validate any new column references
    if _df is not None:
        valid_cols = set(_df.columns.tolist())
        for key in ("x", "y", "color"):
            val = updates.get(key)
            if val and val != "—" and val not in valid_cols:
                return f"⚠️ {key} column '{val}' not found in dataset"
    _ai_dashboard_charts[index].update(updates)
    return f"✅ Chart {index} updated."


def remove_dashboard_chart(index: int) -> str:
    """
    Remove a chart from the AI dashboard by its 0-based index.

    index: 0-based position of the chart to remove

    Returns confirmation with remaining chart count.
    """
    if index < 0 or index >= len(_ai_dashboard_charts):
        return (
            f"⚠️ Invalid chart index {index}. "
            f"Dashboard has {len(_ai_dashboard_charts)} charts."
        )
    removed = _ai_dashboard_charts.pop(index)
    return (
        f"✅ Removed chart '{removed.get('title', index)}'. "
        f"Dashboard now has {len(_ai_dashboard_charts)} charts."
    )
