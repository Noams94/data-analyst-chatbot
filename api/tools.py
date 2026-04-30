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

    Examples:
        "df.groupby('category')['revenue'].sum().sort_values(ascending=False).head(10)"
        "df[df['age'] > 30].describe()"
        "df.corr(numeric_only=True).round(2)"
    """
    s = current()
    s.pending_code_snippets.append({"type": "analysis", "code": pandas_code.strip()})
    local_ns = {"df": s.df.copy(), "pd": pd, "np": np, "json": json}
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
