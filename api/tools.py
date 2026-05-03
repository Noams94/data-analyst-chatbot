"""
Analysis tools registered with the chatlas Chat object.

Per-request state lives in `api.chat_session.current()` — set by the request
handler before chatlas streams. Module-level state from the legacy `tools.py`
has been removed so this module is safe under concurrent FastAPI requests.

The dashboard-builder tools (set/add/update/remove_dashboard_chart) are out of
scope for the MVP and intentionally absent.
"""

from __future__ import annotations

import io
import json
import re
import tempfile
import traceback
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from api.chat_session import current

# ─── Matplotlib style ────────────────────────────────────────────────────────

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
sns.set_style("whitegrid")

_CHART_DPI = 130
_CHART_FIGSIZE = (10, 5)

_MONTH_ORDER: dict[str, int] = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
])}

_HEBREW_RE = re.compile(r"[֐-׿יִ-ﭏ]")


def _contains_hebrew(text: str) -> bool:
    return bool(_HEBREW_RE.search(str(text)))


def _series_has_hebrew(series: pd.Series) -> bool:
    return any(_contains_hebrew(str(v)) for v in series.dropna().head(20))


def _sort_chronological(grp_df: pd.DataFrame, x_col: str):
    sample = [str(v).lower() for v in grp_df[x_col].head(12)]
    if not any(v in _MONTH_ORDER for v in sample):
        return None
    order = grp_df[x_col].apply(lambda v: _MONTH_ORDER.get(str(v).lower(), 99))
    return grp_df.iloc[order.argsort().values].reset_index(drop=True)


# ─── Registered tools ────────────────────────────────────────────────────────

def get_data_overview() -> str:
    """
    Get a comprehensive overview of the loaded dataset.
    Returns: column types, missing values, numeric stats, and sample rows.
    Always call this first before any analysis.
    """
    s = current()
    df = s.df
    buf = io.StringIO()
    buf.write(f"File: {s.data_name}\n")
    buf.write(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n\n")

    rows = []
    for col in df.columns:
        nulls = df[col].isna().sum()
        rows.append({
            "Column": col,
            "Type": str(df[col].dtype),
            "Non-null": f"{len(df) - nulls:,}",
            "Null%": f"{nulls / len(df) * 100:.1f}%",
            "Unique": f"{df[col].nunique():,}",
        })
    buf.write("Column Info:\n")
    buf.write(pd.DataFrame(rows).to_string(index=False))
    buf.write("\n\n")

    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols):
        buf.write("Numeric Summary:\n")
        buf.write(df[num_cols].describe().round(2).to_string())
        buf.write("\n\n")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols):
        buf.write("Top Values per Categorical Column:\n")
        for col in cat_cols[:6]:
            top = df[col].value_counts().head(5)
            buf.write(f"  {col}: {dict(top)}\n")
        buf.write("\n")

    buf.write("Sample (first 5 rows):\n")
    buf.write(df.head(5).to_string(index=False))

    buf.write("\n\nSQL access: query the table named \"df\" using run_sql().\n")
    cols_hint = ", ".join(f'"{c}"' for c in df.columns)
    buf.write(f"Columns: {cols_hint}\n")

    return buf.getvalue()


def run_analysis(pandas_code: str) -> str:
    """
    Run Python/pandas code on the DataFrame (variable name: `df`) and return the result.
    Use for aggregations, groupby, correlations, filtering, pivot tables, etc.

    Available in the namespace: df, pd, np, json, math, re, itertools, scipy.stats (as stats).
    statsmodels.api available as sm if installed.

    Examples:
        "df.groupby('category')['revenue'].sum().sort_values(ascending=False).head(10)"
        "df[df['age'] > 30].describe()"
        "stats.ttest_ind(df[df['group']=='A']['value'], df[df['group']=='B']['value'])"
    """
    import math as _math, re as _re, itertools as _itertools
    import scipy.stats as _scipy_stats
    s = current()
    s.pending_code_snippets.append({"type": "analysis", "code": pandas_code.strip()})
    local_ns: dict = {
        "df": s.df.copy(), "pd": pd, "np": np, "json": json,
        "math": _math, "re": _re, "itertools": _itertools,
        "stats": _scipy_stats,
    }
    try:
        import statsmodels.api as _sm
        local_ns["sm"] = _sm
    except ImportError:
        pass
    try:
        lines = [l for l in pandas_code.strip().splitlines() if l.strip()]
        if not lines:
            return "No code provided."
        if len(lines) > 1:
            exec("\n".join(lines[:-1]), local_ns)
        result = eval(lines[-1], local_ns)
        if isinstance(result, pd.DataFrame):
            return result.to_string(max_rows=50, max_cols=20)
        if isinstance(result, pd.Series):
            return result.to_string(max_rows=50)
        if result is None:
            return "(No output — code executed successfully)"
        return str(result)
    except Exception:
        return f"Error:\n{traceback.format_exc()}"


_SQL_FORBIDDEN = re.compile(
    r"\b(CREATE|DROP|ALTER|INSERT|UPDATE|DELETE|ATTACH|COPY|EXPORT|IMPORT|INSTALL|LOAD|CALL|PRAGMA)\b",
    re.IGNORECASE,
)


def run_sql(sql_query: str) -> str:
    """
    Run a read-only SQL query on the dataset using DuckDB.
    The table name is "df". Quote columns with spaces using double-quotes.

    Examples:
        "SELECT product, SUM(revenue) AS total FROM df GROUP BY product ORDER BY total DESC LIMIT 10"
        "SELECT * FROM df WHERE age > 30"
        "SELECT CORR(price, quantity) FROM df"
    """
    s = current()
    if _SQL_FORBIDDEN.search(sql_query):
        return "⚠️ Only SELECT queries are allowed. DDL/DML statements are blocked."

    s.pending_code_snippets.append({"type": "sql", "code": sql_query.strip()})
    con = duckdb.connect(":memory:")
    try:
        con.register("df", s.df)
        result = con.execute(sql_query).fetchdf()
    except Exception as e:
        return f"SQL Error: {e}"
    finally:
        con.close()
    if result.empty:
        return "(Query returned no rows)"
    return result.to_string(max_rows=50, max_cols=20, index=False)


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

    chart_type: bar | barh | line | scatter | hist | box | pie | heatmap | count
    aggregation: sum | mean | count | median | max | min
    top_n: if > 0, keep only top N categories
    """
    s = current()
    df = s.df.copy()
    fig, ax = plt.subplots(figsize=_CHART_FIGSIZE, dpi=_CHART_DPI)
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

            xv_str = [str(v) for v in xv]
            if t == "line":
                ax.plot(range(len(xv_str)), yv, marker="o", linewidth=2, markersize=5)
                ax.fill_between(range(len(xv_str)), yv, alpha=0.1)
                ax.set_xticks(range(len(xv_str)))
                ax.set_xticklabels(xv_str, rotation=45, ha="right")
                ax.set_ylabel(ylabel)
            elif t == "barh":
                ax.barh(range(len(xv_str)), yv)
                ax.set_yticks(range(len(xv_str)))
                ax.set_yticklabels(xv_str, fontsize=9)
                ax.set_xlabel(ylabel)
            else:
                ax.bar(range(len(xv_str)), yv)
                ax.set_xticks(range(len(xv_str)))
                ax.set_xticklabels(xv_str, rotation=45, ha="right", fontsize=9)
                ax.set_ylabel(ylabel)

            if t != "barh":
                ax.set_xlabel(x_column)
            ax.set_title(title or f"{ylabel} by {x_column}")

        elif t == "scatter":
            kw = {"alpha": 0.55, "s": 35, "edgecolors": "none"}
            if color_column and color_column in df.columns:
                for val, grp in df.groupby(color_column):
                    ax.scatter(grp[x_column], grp[y_column], label=str(val), **kw)
                ax.legend(title=color_column, bbox_to_anchor=(1.02, 1),
                          loc="upper left", fontsize=8, title_fontsize=8)
            else:
                ax.scatter(df[x_column], df[y_column], **kw)
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_title(title or f"{y_column} vs {x_column}")

        elif t == "box":
            if x_column and x_column in df.columns:
                groups = df[x_column].unique()
                data = [df[df[x_column] == g][y_column].dropna() for g in groups]
                ax.boxplot(data, labels=[str(g) for g in groups], patch_artist=True)
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

        # Save to a per-request temp file. The API layer uploads to Blob and
        # rewrites the path to a public URL when emitting the SSE `chart` event.
        safe = "".join(c for c in (title or f"{t}_{x_column}") if c.isalnum() or c in "_-")[:48] or "chart"
        tmp = tempfile.NamedTemporaryFile(prefix=f"{safe}_", suffix=".png", delete=False)
        path = Path(tmp.name)
        tmp.close()
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        s.pending_charts.append(path)
        s.pending_chart_configs.append({
            "type": "image",
            "path": str(path),
            "title": title or f"{t} – {x_column}",
            "chart_type": t,
        })
        code = (
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
        s.pending_code_snippets.append({"type": "chart", "code": code})

        return f"✅ Chart created: '{title or t}' — displayed below."

    except Exception:
        plt.close(fig)
        return f"Chart error:\n{traceback.format_exc()}"


def _build_plotly_figure(
    chart_type: str,
    x_column: str,
    y_column: str = "",
    title: str = "",
    color_column: str = "",
    aggregation: str = "sum",
    top_n: int = 0,
    bins: int = 20,
    sort: bool = True,
    df: "pd.DataFrame | None" = None,
) -> "tuple[object, str]":
    """
    Build and return (plotly_figure, code_snippet_str).
    Raises ValueError for unknown chart_type, any other exception for data errors.
    `df` defaults to current().df.copy() when None.
    """
    import plotly.express as px

    if df is None:
        df = current().df.copy()

    t = chart_type.lower().strip()
    fig = None

    if t == "heatmap":
        num_df = df.select_dtypes(include="number")
        corr = num_df.corr().round(3)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title=title or "Correlation Heatmap",
            aspect="auto",
        )

    elif t == "hist":
        fig = px.histogram(
            df, x=x_column, nbins=bins,
            title=title or f"Distribution of {x_column}",
            labels={x_column: x_column},
        )

    elif t == "scatter":
        kw: dict = {"x": x_column, "y": y_column, "opacity": 0.6,
                    "title": title or f"{y_column} vs {x_column}"}
        if color_column and color_column in df.columns:
            kw["color"] = color_column
        fig = px.scatter(df, **kw)

    elif t == "box":
        kw = {"y": y_column, "title": title or f"Distribution of {y_column}"}
        if x_column and x_column in df.columns:
            kw["x"] = x_column
        fig = px.box(df, **kw)

    elif t == "pie":
        if y_column:
            grp = df.groupby(x_column)[y_column].agg(aggregation or "sum").reset_index()
            vals_col = y_column
        else:
            grp = df[x_column].value_counts().reset_index()
            grp.columns = [x_column, "count"]
            vals_col = "count"
        if top_n > 0:
            grp = grp.nlargest(top_n, vals_col)
        fig = px.pie(grp, names=x_column, values=vals_col,
                     title=title or f"Distribution of {x_column}")

    elif t in ("bar", "barh", "line"):
        agg = aggregation or "sum"
        if y_column:
            grp = df.groupby(x_column)[y_column].agg(agg).reset_index()
            grp.columns = [x_column, y_column]
            chr_grp = _sort_chronological(grp, x_column)
            if chr_grp is not None:
                grp = chr_grp
            elif sort:
                grp = grp.sort_values(y_column, ascending=(t == "barh"))
            if top_n > 0:
                grp = grp.tail(top_n) if t == "barh" else grp.head(top_n)
            xv, yv, ylabel = grp[x_column], grp[y_column], f"{agg}({y_column})"
        else:
            vc = df[x_column].value_counts()
            if sort:
                vc = vc.sort_values(ascending=(t == "barh"))
            if top_n > 0:
                vc = vc.tail(top_n) if t == "barh" else vc.head(top_n)
            xv, yv, ylabel = vc.index.tolist(), vc.values.tolist(), "Count"

        if t == "line":
            fig = px.line(x=xv, y=yv, title=title or f"{ylabel} by {x_column}",
                          labels={"x": x_column, "y": ylabel}, markers=True)
        elif t == "barh":
            fig = px.bar(x=yv, y=xv, orientation="h",
                         title=title or f"{ylabel} by {x_column}",
                         labels={"x": ylabel, "y": x_column})
        else:
            fig = px.bar(x=xv, y=yv, title=title or f"{ylabel} by {x_column}",
                         labels={"x": x_column, "y": ylabel})

        # RTL: reverse the categorical axis when labels are Hebrew
        x_sample = list(xv)[:20] if hasattr(xv, "__iter__") else []
        if any(_contains_hebrew(str(v)) for v in x_sample):
            if t == "barh":
                fig.update_layout(yaxis=dict(autorange="reversed"))
            else:
                fig.update_layout(xaxis=dict(autorange="reversed"))
    else:
        raise ValueError(
            f"Unknown chart_type: '{chart_type}'. "
            "Valid: bar, barh, line, scatter, hist, box, pie, heatmap"
        )

    fig.update_layout(
        template="plotly_white",
        font_family="sans-serif",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )

    code = (
        f"create_interactive_chart(\n"
        f"    chart_type={chart_type!r},\n"
        f"    x_column={x_column!r},\n"
        + (f"    y_column={y_column!r},\n" if y_column else "")
        + (f"    title={title!r},\n" if title else "")
        + ")"
    )
    return fig, code


def create_interactive_chart(
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
    Create an interactive chart (Plotly) — user can zoom, pan, hover, and download.

    chart_type: bar | barh | line | scatter | hist | box | pie | heatmap
    aggregation: sum | mean | count | median | max | min
    top_n: if > 0, keep only top N categories
    Prefer this over create_chart for all visualizations.
    """
    s = current()
    try:
        fig, code = _build_plotly_figure(
            chart_type, x_column, y_column, title,
            color_column, aggregation, top_n, bins, sort,
        )
    except ValueError as e:
        return str(e)
    except Exception:
        return f"Chart error:\n{traceback.format_exc()}"

    spec = fig.to_json()
    s.pending_plotly_charts.append({"spec": spec, "title": title or f"{chart_type} – {x_column}"})
    s.pending_code_snippets.append({"type": "chart", "code": code})
    return f"✅ Interactive chart created: '{title or chart_type}' — displayed below."


def add_to_dashboard(
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
    Build a Plotly chart and add it to this chat's persistent dashboard.

    The dashboard shows all added charts in a 2-column interactive grid,
    accessible via the Dashboard button in the chat header. Use this when
    the user asks for a dashboard, overview panel, summary view, or wants
    multiple charts together. Call once per chart (2–6 charts is ideal).

    chart_type: bar | barh | line | scatter | hist | box | pie | heatmap
    aggregation: sum | mean | count | median | max | min
    top_n: if > 0, keep only top N categories
    """
    s = current()
    try:
        fig, _ = _build_plotly_figure(
            chart_type, x_column, y_column, title,
            color_column, aggregation, top_n, bins, sort,
        )
    except ValueError as e:
        return str(e)
    except Exception:
        return f"Chart error:\n{traceback.format_exc()}"

    slot = len(s.pending_dashboard_charts)
    spec = fig.to_json()
    chart_title = title or f"{chart_type} – {x_column}"
    s.pending_dashboard_charts.append({"spec": spec, "title": chart_title, "position": slot})
    return f"✅ '{chart_title}' added to dashboard (slot {slot + 1})."


def clear_dashboard_tool() -> str:
    """
    Remove all charts from this chat's dashboard.
    Use when the user says 'reset the dashboard', 'start over', or 'clear the dashboard'.
    """
    from api import state as _state
    s = current()
    count = _state.clear_dashboard(s.chat_id)
    s.pending_dashboard_charts.clear()
    return f"✅ Dashboard cleared ({count} chart{'s' if count != 1 else ''} removed)."


def detect_outliers(column: str, method: str = "iqr") -> str:
    """
    Detect outliers in a numeric column using IQR or z-score method.

    method: "iqr" (default) — flags rows below Q1−1.5×IQR or above Q3+1.5×IQR
            "zscore" — flags rows where |z-score| > 3

    Returns count, percentage, value range of outliers, and up to 5 example rows.
    """
    s = current()
    df = s.df
    if column not in df.columns:
        return f"Column '{column}' not found. Available columns: {', '.join(df.columns)}"
    col = df[column].dropna()
    if not pd.api.types.is_numeric_dtype(col):
        return f"Column '{column}' is not numeric (dtype: {df[column].dtype}). Outlier detection requires a numeric column."

    if method == "zscore":
        z = (col - col.mean()) / col.std()
        mask = z.abs() > 3
        label = "z-score > 3"
    else:
        q1, q3 = col.quantile(0.25), col.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (col < lower) | (col > upper)
        label = f"IQR bounds [{lower:.4g}, {upper:.4g}]"

    outlier_idx = mask[mask].index
    count = len(outlier_idx)
    total = len(col)
    pct = count / total * 100

    lines = [
        f"Outlier detection — column: '{column}', method: {method} ({label})",
        f"Found {count} outliers out of {total} non-null values ({pct:.2f}%)",
    ]
    if count > 0:
        outlier_vals = df.loc[outlier_idx, column]
        lines.append(f"Outlier range: min={outlier_vals.min():.4g}, max={outlier_vals.max():.4g}")
        lines.append(f"Non-outlier range: min={col[~mask].min():.4g}, max={col[~mask].max():.4g}")
        sample = df.loc[outlier_idx[:5], [column]]
        lines.append(f"\nTop 5 outlier rows:\n{sample.to_string()}")
    else:
        lines.append("No outliers detected.")
    return "\n".join(lines)


def compute_statistics(column1: str, column2: str = "", test: str = "auto") -> str:
    """
    Compute descriptive statistics and/or a statistical test.

    One column (numeric): skewness, kurtosis, normality test (Shapiro-Wilk or D'Agostino).
    Two numeric columns: Pearson + Spearman correlation with p-values.
    Numeric + categorical: one-way ANOVA across categorical groups.
    Two categorical: chi-squared test of independence.

    test: "auto" selects the appropriate test based on dtypes.
    Interprets results at α = 0.05.
    """
    import scipy.stats as _stats
    s = current()
    df = s.df

    def _check(col: str) -> str | None:
        if col not in df.columns:
            return f"Column '{col}' not found. Available: {', '.join(df.columns)}"
        return None

    err = _check(column1)
    if err:
        return err
    if column2:
        err = _check(column2)
        if err:
            return err

    col1 = df[column1].dropna()
    is_num1 = pd.api.types.is_numeric_dtype(col1)

    lines: list[str] = []

    if not column2:
        # Single column: descriptive + normality
        if not is_num1:
            vc = df[column1].value_counts()
            return (
                f"Column '{column1}' is categorical (dtype: {df[column1].dtype}).\n"
                f"Value counts (top 10):\n{vc.head(10).to_string()}\n"
                f"Unique values: {df[column1].nunique()}"
            )
        n = len(col1)
        lines.append(f"Descriptive statistics for '{column1}' (n={n:,}):")
        lines.append(f"  mean={col1.mean():.4g}, std={col1.std():.4g}")
        lines.append(f"  min={col1.min():.4g}, Q1={col1.quantile(.25):.4g}, median={col1.median():.4g}, Q3={col1.quantile(.75):.4g}, max={col1.max():.4g}")
        lines.append(f"  skewness={col1.skew():.4f}  (>1 or <−1 = strongly skewed)")
        lines.append(f"  kurtosis={col1.kurtosis():.4f}  (>3 = heavy tails vs normal)")

        if n <= 5000:
            stat, p = _stats.shapiro(col1)
            test_name = "Shapiro-Wilk"
        else:
            stat, p = _stats.normaltest(col1)
            test_name = "D'Agostino K²"
        conclusion = "likely normal (fail to reject H₀)" if p > 0.05 else "NOT normal (reject H₀)"
        lines.append(f"\n{test_name} normality test: W={stat:.4f}, p={p:.4g} → {conclusion} at α=0.05")
        return "\n".join(lines)

    col2 = df[column2].dropna()
    is_num2 = pd.api.types.is_numeric_dtype(col2)

    if is_num1 and is_num2:
        # Correlation
        common = df[[column1, column2]].dropna()
        n = len(common)
        c1, c2 = common[column1], common[column2]
        r_p, p_p = _stats.pearsonr(c1, c2)
        r_s, p_s = _stats.spearmanr(c1, c2)
        lines.append(f"Correlation analysis: '{column1}' vs '{column2}' (n={n:,})")
        lines.append(f"  Pearson r  = {r_p:.4f}  (p={p_p:.4g})")
        lines.append(f"  Spearman ρ = {r_s:.4f}  (p={p_s:.4g})")
        sig = "statistically significant" if min(p_p, p_s) < 0.05 else "NOT statistically significant"
        strength = abs(r_p)
        label = "negligible" if strength < 0.1 else "weak" if strength < 0.3 else "moderate" if strength < 0.5 else "strong"
        lines.append(f"  → {sig} {label} {'positive' if r_p >= 0 else 'negative'} linear relationship at α=0.05")

    elif is_num1 and not is_num2:
        # ANOVA: numeric by category
        groups = [g[column1].dropna().values for _, g in df.groupby(column2)]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            return f"Need ≥2 groups in '{column2}' for ANOVA."
        f_stat, p = _stats.f_oneway(*groups)
        n_total = sum(len(g) for g in groups)
        lines.append(f"One-way ANOVA: '{column1}' across groups of '{column2}'")
        lines.append(f"  Groups: {df[column2].nunique()} categories, n={n_total:,} total")
        lines.append(f"  F={f_stat:.4f}, p={p:.4g}")
        sig = "statistically significant" if p < 0.05 else "NOT statistically significant"
        lines.append(f"  → {sig} difference in means across groups at α=0.05")
        # Group means
        means = df.groupby(column2)[column1].mean().sort_values(ascending=False)
        lines.append(f"\nGroup means:\n{means.round(4).to_string()}")

    elif not is_num1 and not is_num2:
        # Chi-squared
        ct = pd.crosstab(df[column1], df[column2])
        chi2, p, dof, expected = _stats.chi2_contingency(ct)
        n = ct.values.sum()
        lines.append(f"Chi-squared test of independence: '{column1}' vs '{column2}'")
        lines.append(f"  n={n:,}, degrees of freedom={dof}")
        lines.append(f"  χ²={chi2:.4f}, p={p:.4g}")
        sig = "statistically significant" if p < 0.05 else "NOT statistically significant"
        lines.append(f"  → {sig} association between the two variables at α=0.05")
        # Cramér's V (effect size)
        v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))
        lines.append(f"  Cramér's V (effect size) = {v:.4f}  (0=none, 0.1=small, 0.3=medium, 0.5+=large)")
    else:
        lines.append(f"Cannot compute statistics: '{column1}' is {'numeric' if is_num1 else 'categorical'}, '{column2}' is {'numeric' if is_num2 else 'categorical'}. Try swapping the column order.")

    return "\n".join(lines)


def create_map(
    lat_column: str,
    lng_column: str,
    label_column: str = "",
    value_column: str = "",
    title: str = "",
    zoom: int = 4,
) -> str:
    """
    Create an interactive map from the dataset and display it in the UI.

    lat_column: column with latitude values (numeric, degrees)
    lng_column: column with longitude values (numeric, degrees)
    label_column: column to show as hover label (optional)
    value_column: column to use for color-coding points (optional)
    zoom: initial zoom level 1–18 (default 4 — country/continent view)

    Use get_data_overview() first to identify which columns hold coordinates.
    Common column names: lat/latitude/קו_רוחב, lon/lng/longitude/קו_אורך.
    """
    import plotly.express as px

    s = current()
    df = s.df.copy()

    for col in [lat_column, lng_column]:
        if col not in df.columns:
            return f"Column '{col}' not found. Available: {', '.join(df.columns)}"

    keep = [lat_column, lng_column]
    if label_column and label_column in df.columns:
        keep.append(label_column)
    if value_column and value_column in df.columns:
        keep.append(value_column)

    map_df = df[keep].dropna(subset=[lat_column, lng_column])
    if map_df.empty:
        return "No rows with valid lat/lng coordinates found."

    hover_name = label_column if label_column and label_column in df.columns else None
    color = value_column if value_column and value_column in df.columns else None

    try:
        fig = px.scatter_mapbox(
            map_df,
            lat=lat_column,
            lon=lng_column,
            hover_name=hover_name,
            color=color,
            title=title or "Map",
            zoom=zoom,
            height=520,
            mapbox_style="open-street-map",
        )
        fig.update_layout(margin={"l": 0, "r": 0, "t": 40, "b": 0})
    except Exception:
        # Fallback to scatter_geo (no tile server needed)
        fig = px.scatter_geo(
            map_df,
            lat=lat_column,
            lon=lng_column,
            hover_name=hover_name,
            color=color,
            title=title or "Map",
        )
        fig.update_layout(template="plotly_white")

    spec = fig.to_json()
    chart_title = title or "Map"
    s.pending_plotly_charts.append({"spec": spec, "title": chart_title})
    s.pending_code_snippets.append({
        "type": "chart",
        "code": (
            f"create_map(\n"
            f"    lat_column={lat_column!r},\n"
            f"    lng_column={lng_column!r},\n"
            + (f"    label_column={label_column!r},\n" if label_column else "")
            + (f"    value_column={value_column!r},\n" if value_column else "")
            + (f"    title={title!r},\n" if title else "")
            + ")"
        ),
    })
    return f"✅ Map created with {len(map_df):,} points — displayed below."


def compute_nps(rating_column: str) -> str:
    """
    Compute Net Promoter Score (NPS) from a column of 0–10 ratings.

    Detractors: 0–6 | Passives: 7–8 | Promoters: 9–10
    NPS = round(Promoters% − Detractors%)

    Displays a segmented horizontal bar chart plus a score summary.
    Use when the dataset contains customer satisfaction or loyalty ratings.
    """
    import plotly.graph_objects as go

    s = current()
    df = s.df

    if rating_column not in df.columns:
        return f"Column '{rating_column}' not found. Available: {', '.join(df.columns)}"

    col = pd.to_numeric(df[rating_column], errors="coerce").dropna()
    valid = col[col.between(0, 10)]
    if valid.empty:
        return f"No valid 0–10 ratings found in '{rating_column}'."

    n = len(valid)
    detractors = int((valid <= 6).sum())
    passives   = int(((valid >= 7) & (valid <= 8)).sum())
    promoters  = int((valid >= 9).sum())

    det_pct = detractors / n * 100
    pas_pct = passives   / n * 100
    pro_pct = promoters  / n * 100
    score = round(pro_pct - det_pct)

    score_color = "#16a34a" if score >= 50 else "#0891b2" if score >= 0 else "#dc2626"

    fig = go.Figure()
    for name, pct, count, color in [
        (f"Detractors 0–6 ({det_pct:.1f}%)", det_pct, detractors, "#dc2626"),
        (f"Passives 7–8 ({pas_pct:.1f}%)",   pas_pct, passives,   "#eab308"),
        (f"Promoters 9–10 ({pro_pct:.1f}%)", pro_pct, promoters,  "#16a34a"),
    ]:
        fig.add_trace(go.Bar(
            name=name,
            x=[pct], y=["NPS"],
            orientation="h",
            marker_color=color,
            text=f"{pct:.1f}%",
            textposition="inside",
            hovertemplate=f"{name.split('(')[0].strip()}: {count:,} ({pct:.1f}%)<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        title=dict(
            text=f"NPS Score: <b>{score:+d}</b>   (n={n:,})",
            font=dict(size=18, color=score_color),
        ),
        template="plotly_white",
        xaxis=dict(title="Percentage (%)", range=[0, 100]),
        yaxis=dict(showticklabels=False),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=220,
    )

    spec = fig.to_json()
    chart_title = f"NPS Score: {score:+d}"
    s.pending_plotly_charts.append({"spec": spec, "title": chart_title})
    s.pending_code_snippets.append({
        "type": "analysis",
        "code": f"compute_nps(rating_column={rating_column!r})",
    })

    return "\n".join([
        f"NPS Analysis — column: '{rating_column}' (n={n:,})",
        f"  Score:      {score:+d}",
        f"  Promoters:  {promoters:,}  ({pro_pct:.1f}%)",
        f"  Passives:   {passives:,}  ({pas_pct:.1f}%)",
        f"  Detractors: {detractors:,}  ({det_pct:.1f}%)",
        "",
        "Chart displayed below.",
    ])


def suggest_next_analyses() -> str:
    """Examine the dataset and suggest meaningful follow-up analyses."""
    df = current().df
    suggestions: list[str] = []
    num_cols = list(df.select_dtypes(include="number").columns)
    cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)
    date_cols = [c for c in df.columns
                 if "date" in c.lower() or "time" in c.lower()
                 or pd.api.types.is_datetime64_any_dtype(df[c])]

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

    missing = df.isna().sum()
    if missing.sum() > 0:
        top = missing[missing > 0].sort_values(ascending=False).head(3)
        suggestions.append(f"🔎 Missing value analysis: {dict(top)}")

    if num_cols:
        col = num_cols[0]
        outliers = ((df[col] - df[col].mean()).abs() > 3 * df[col].std()).sum()
        if outliers > 0:
            suggestions.append(f"⚠️ Potential outliers in '{col}': {outliers} rows")

    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(suggestions))
