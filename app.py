"""
🤖 Data Analyst Chatbot — Streamlit App
Bilingual (Hebrew / English) interface powered by chatlas + Claude Opus 4.6
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
from chatlas import ChatAnthropic
from chatlas._turn import ContentToolRequest, ContentToolResult

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import tools
import plotly.express as px
import plotly.graph_objects as go

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Analyst Bot | צ'אטבוט נתונים",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── i18n ─────────────────────────────────────────────────────────────────────
TEXT = {
    "he": {
        "page_title":       "🤖 צ'אטבוט ניתוח נתונים",
        "page_sub":         "שאל אותי כל שאלה על הנתונים שלך",
        "lang_btn":         "🇬🇧 English",
        "api_header":       "🔑 מפתח API",
        "api_label":        "Anthropic API Key",
        "api_placeholder":  "sk-ant-...",
        "api_help":         "המפתח נשמר בזיכרון בלבד ונמחק בסגירת הדפדפן",
        "api_set":          "✅ מפתח מוגדר",
        "api_from_env":     "✅ מפתח מוגדר (משתנה סביבה)",
        "api_missing":      "❌ נדרש מפתח API",
        "api_save":         "שמור",
        "api_clear":        "נקה",
        "api_hint":         "קבל מפתח בחינם: console.anthropic.com",
        "upload_header":    "📂 העלאת קובץ",
        "upload_label":     "גרור קובץ נתונים לכאן",
        "upload_help":      "CSV, Excel, JSON, Parquet",
        "demo_btn":         "🎲 טען דאטה לדוגמה",
        "demo_loaded":      "✅ נטענה דוגמת מכירות — 500 שורות",
        "data_header":      "📊 פרטי הדאטה",
        "rows":             "שורות",
        "cols":             "עמודות",
        "no_data_warn":     "⬆️ העלה קובץ נתונים להתחלה",
        "quick_header":     "💡 שאלות מהירות",
        "quick_qs": [
            "תראה לי סקירה של הנתונים",
            "מה המוצרים הכי נמכרים?",
            "הצג גרף הכנסות לפי חודש",
            "מה הקורלציות בין העמודות?",
            "תציע ניתוחים נוספים",
        ],
        "clear_btn":        "🗑️ נקה שיחה",
        "input_placeholder":"שאל שאלה על הנתונים... (עברית / English)",
        "thinking":         "מנתח...",
        "welcome": (
            "שלום! אני הצ'אטבוט לניתוח נתונים שלך. 👋\n\n"
            "אני יכול לעזור לך:\n"
            "- 📊 לנתח ולסכם נתונים\n"
            "- 📈 ליצור גרפים ויזואליים\n"
            "- 🔍 לגלות תובנות ומגמות\n"
            "- 💡 להציע ניתוחים נוספים\n\n"
            "**התחל בהעלאת קובץ נתונים** (CSV, Excel, JSON) או לחץ על 'טען דאטה לדוגמה'."
        ),
        "tool_badge":       "🔧 כלי:",
        # לשוניות
        "tab_chat":           "🤖 AI Chat",
        "tab_charts":         "📊 גרפים",
        "tab_dashboard":      "📈 דשבורד",
        "tab_data":           "🗂 נתונים",
        # בונה גרפים
        "chart_type":         "סוג גרף",
        "chart_x":            "ציר X",
        "chart_y":            "ציר Y (מספרי)",
        "chart_color_by":     "צבע לפי (אופציונלי)",
        "chart_palette":      "🎨 פלטת צבעים",
        "chart_title_lbl":    "כותרת הגרף",
        "chart_title_ph":     "כותרת אופציונלית...",
        "chart_sample_lbl":   "🔢 גודל מדגם לגרף",
        "chart_corr_method":  "שיטת קורלציה",
        "add_to_dash":        "📌 הוסף גרף זה לדשבורד",
        "chart_added":        "✅ גרף נוסף לדשבורד!",
        "no_charts_hint":     "📊 עבור ללשונית **גרפים**, בנה גרף ולחץ **'הוסף לדשבורד'** כדי להתחיל.",
        "chart_sample_note":  "הגרף מציג {n:,} שורות מתוך {total:,}.",
        # דשבורד
        "dash_charts_count":  "גרפים בדשבורד",
        "dash_layout":        "פריסה",
        "dash_clear":         "🗑 נקה דשבורד",
        "dash_col_suffix":    "עמ'",
        "dash_remove":        "× הסר",
        # תצוגת נתונים
        "data_title":         "📋 תצוגת נתונים",
        "data_search":        "🔍 חיפוש חופשי",
        "data_search_ph":     "הקלד ערך לחיפוש...",
        "data_filter_col":    "סנן לפי עמודה",
        "data_filter_all":    "הכל",
        "data_stats_chk":     "הצג סטטיסטיקות מפורטות",
        "data_autofix_btn":   "🔧 תקן אוטומטית",
        "data_fixed":         "✅ הנתונים תוקנו אוטומטית",
        "data_warnings_hdr":  "אזהרות איכות נתונים",
    },
    "en": {
        "page_title":       "🤖 Data Analyst Chatbot",
        "page_sub":         "Ask me anything about your data",
        "lang_btn":         "🇮🇱 עברית",
        "api_header":       "🔑 API Key",
        "api_label":        "Anthropic API Key",
        "api_placeholder":  "sk-ant-...",
        "api_help":         "Stored in memory only — cleared when you close the browser",
        "api_set":          "✅ Key configured",
        "api_from_env":     "✅ Key configured (env variable)",
        "api_missing":      "❌ API key required",
        "api_save":         "Save",
        "api_clear":        "Clear",
        "api_hint":         "Get a free key at: console.anthropic.com",
        "upload_header":    "📂 Upload File",
        "upload_label":     "Drop a data file here",
        "upload_help":      "CSV, Excel, JSON, Parquet",
        "demo_btn":         "🎲 Load Sample Data",
        "demo_loaded":      "✅ Sample sales data loaded — 500 rows",
        "data_header":      "📊 Dataset Info",
        "rows":             "Rows",
        "cols":             "Columns",
        "no_data_warn":     "⬆️ Upload a data file to get started",
        "quick_header":     "💡 Quick Questions",
        "quick_qs": [
            "Give me a data overview",
            "What are the top-selling products?",
            "Show a revenue chart by month",
            "What are the correlations?",
            "Suggest further analyses",
        ],
        "clear_btn":        "🗑️ Clear Chat",
        "input_placeholder":"Ask a question about your data...",
        "thinking":         "Analyzing...",
        "welcome": (
            "Hello! I'm your Data Analyst Chatbot. 👋\n\n"
            "I can help you:\n"
            "- 📊 Analyze and summarize data\n"
            "- 📈 Create visualizations\n"
            "- 🔍 Discover insights and trends\n"
            "- 💡 Suggest further analyses\n\n"
            "**Start by uploading a data file** (CSV, Excel, JSON) "
            "or click 'Load Sample Data'."
        ),
        "tool_badge":       "🔧 Tool:",
        # Tabs
        "tab_chat":           "🤖 AI Chat",
        "tab_charts":         "📊 Charts",
        "tab_dashboard":      "📈 Dashboard",
        "tab_data":           "🗂 Data",
        # Chart builder
        "chart_type":         "Chart Type",
        "chart_x":            "X Axis",
        "chart_y":            "Y Axis (numeric)",
        "chart_color_by":     "Color by (optional)",
        "chart_palette":      "🎨 Color Palette",
        "chart_title_lbl":    "Chart Title",
        "chart_title_ph":     "Optional title...",
        "chart_sample_lbl":   "🔢 Sample Size",
        "chart_corr_method":  "Correlation Method",
        "add_to_dash":        "📌 Add chart to Dashboard",
        "chart_added":        "✅ Chart added to Dashboard!",
        "no_charts_hint":     "📊 Go to the **Charts** tab, build a chart, then click **'Add to Dashboard'**.",
        "chart_sample_note":  "Chart shows {n:,} of {total:,} rows.",
        # Dashboard
        "dash_charts_count":  "charts in dashboard",
        "dash_layout":        "Layout",
        "dash_clear":         "🗑 Clear Dashboard",
        "dash_col_suffix":    "col",
        "dash_remove":        "× Remove",
        # Data view
        "data_title":         "📋 Data View",
        "data_search":        "🔍 Free Search",
        "data_search_ph":     "Type a value to search...",
        "data_filter_col":    "Filter by column",
        "data_filter_all":    "All",
        "data_stats_chk":     "Show detailed statistics",
        "data_autofix_btn":   "🔧 Auto-fix",
        "data_fixed":         "✅ Data auto-fixed",
        "data_warnings_hdr":  "data quality warnings",
    },
}

SYSTEM_PROMPT = """
You are an expert, bilingual Data Analyst assistant.
Respond in the same language the user writes in (Hebrew or English).

## Capabilities
- Use get_data_overview() first to understand the dataset
- Use run_analysis() for pandas computations (groupby, corr, filter, etc.)
- Use create_chart() to visualize findings (ALWAYS visualize key insights)
- Use suggest_next_analyses() to propose follow-up ideas

## Response Format
1. 📊 **Key Findings** — bullet points with the main numbers
2. 🔍 **Interpretation** — what the numbers mean
3. 💡 **Insight / Recommendation** — business or analytical conclusion
4. ➡️ **Follow-up** — 1–2 natural follow-up questions

## Chart Guidelines
- barh  → many/long category names
- line  → time series or ordered x-axis
- scatter → two numeric variables
- heatmap → correlations
- hist → distributions
- Always set a clear, descriptive title in the user's language

## Style
- Be concise yet complete
- Round numbers to 2 decimal places
- When writing Hebrew, keep markdown formatting clean
""".strip()

# ─── Data Validation ──────────────────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame) -> list:
    """Return list of data-quality warnings (numeric-as-string, unparsed dates)."""
    import re as _re
    warnings_list = []
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(30).astype(str)
        if len(sample) == 0:
            continue
        numeric_hits = sample.str.match(r"^[\d,\.]+%?$").sum()
        if numeric_hits >= len(sample) * 0.7:
            warnings_list.append(
                f"עמודה **'{col}'** נראית מספרית אך מאוחסנת כטקסט "
                f"(לדוגמה: `{sample.iloc[0]}`). ניתן לתקן אוטומטית."
            )
            continue
        date_hits = sample.str.match(r"^\d{1,4}[-/\.]\d{1,2}[-/\.]\d{2,4}$").sum()
        if date_hits >= len(sample) * 0.7:
            warnings_list.append(
                f"עמודה **'{col}'** נראית כתאריך אך לא פוענחה כ-datetime "
                f"(לדוגמה: `{sample.iloc[0]}`). ניתן לתקן אוטומטית."
            )
    return warnings_list


def auto_fix_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-convert numeric-as-string and date columns."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(30).astype(str)
        if len(sample) == 0:
            continue
        numeric_hits = sample.str.match(r"^[\d,\.]+%?$").sum()
        if numeric_hits >= len(sample) * 0.7:
            try:
                df[col] = (
                    df[col].astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("%", "", regex=False)
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")
                continue
            except Exception:
                pass
        date_hits = sample.str.match(r"^\d{1,4}[-/\.]\d{1,2}[-/\.]\d{2,4}$").sum()
        if date_hits >= len(sample) * 0.7:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df


# ─── Chart Builder (Plotly) ───────────────────────────────────────────────────

COLOR_PALETTES = {
    "סגול / Purple": {"single": "#7c6af7", "seq": px.colors.sequential.Purp,    "scale": "Purp"},
    "כחול / Blue":   {"single": "#1a73e8", "seq": px.colors.sequential.Blues,   "scale": "Blues"},
    "ירוק / Green":  {"single": "#2ecc71", "seq": px.colors.sequential.Greens,  "scale": "Greens"},
    "כתום / Orange": {"single": "#f39c12", "seq": px.colors.sequential.Oranges, "scale": "Oranges"},
    "ורוד / Pink":   {"single": "#f76a8a", "seq": px.colors.sequential.RdPu,    "scale": "RdPu"},
    "ציאן / Teal":   {"single": "#00bcd4", "seq": px.colors.sequential.Teal,    "scale": "Teal"},
}

CHART_TYPES = [
    "עמודות / Bar", "קו / Line", "שטח / Area", "עוגה / Pie",
    "היסטוגרמה / Histogram", "פיזור / Scatter", "Box Plot", "Heatmap",
]


def apply_chart_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="#f8f9fc",
        font_family="Segoe UI, Helvetica Neue, Arial, sans-serif",
        font_color="#1a1a2e",
        title_font_color="#1a1a2e",
    )
    return fig


def build_chart(config: dict, df: pd.DataFrame, sample_size: int = 500) -> go.Figure:
    """Build a Plotly figure from a chart-config dict."""
    chart_type   = config.get("type", CHART_TYPES[0])
    x_col        = config.get("x")
    y_col        = config.get("y", "—")
    color        = config.get("color")
    title        = config.get("title", "")
    palette_name = config.get("palette", list(COLOR_PALETTES.keys())[0])
    corr_method  = config.get("corr_method", "pearson")
    palette      = COLOR_PALETTES.get(palette_name, list(COLOR_PALETTES.values())[0])

    num_cols = df.select_dtypes(include="number").columns.tolist()
    common   = {"title": title} if title else {}
    single   = palette["single"]
    seq      = palette["seq"]
    dfs      = df.head(sample_size)

    # Extract base Hebrew type (handles "עמודות / Bar" → "עמודות" and "Box Plot" → "Box Plot")
    t = chart_type.split(" / ")[0]

    try:
        if t == "עמודות":
            if y_col and y_col != "—":
                fig = px.bar(dfs, x=x_col, y=y_col, color=color,
                             color_discrete_sequence=seq, **common)
            else:
                counts = df[x_col].value_counts().head(20).reset_index()
                counts.columns = [x_col, "count"]
                fig = px.bar(counts, x=x_col, y="count",
                             color_discrete_sequence=[single], **common)

        elif t == "קו":
            y = y_col if y_col and y_col != "—" else (num_cols[0] if num_cols else x_col)
            fig = px.line(dfs, x=x_col, y=y, color=color,
                          color_discrete_sequence=[single], **common)

        elif t == "שטח":
            y = y_col if y_col and y_col != "—" else (num_cols[0] if num_cols else x_col)
            fig = px.area(dfs, x=x_col, y=y, color=color,
                          color_discrete_sequence=[single], **common)

        elif t == "עוגה":
            counts = df[x_col].value_counts().head(10).reset_index()
            counts.columns = [x_col, "count"]
            fig = px.pie(counts, names=x_col, values="count",
                         color_discrete_sequence=seq, **common)

        elif t == "היסטוגרמה":
            col = y_col if y_col and y_col != "—" else (num_cols[0] if num_cols else x_col)
            fig = px.histogram(df, x=col, color_discrete_sequence=[single], **common)

        elif t == "פיזור":
            y = (y_col if y_col and y_col != "—"
                 else (num_cols[1] if len(num_cols) > 1 else (num_cols[0] if num_cols else x_col)))
            fig = px.scatter(dfs, x=x_col, y=y, color=color,
                             color_discrete_sequence=[single], **common)

        elif t == "Box Plot":
            y     = y_col if y_col and y_col != "—" else (num_cols[0] if num_cols else x_col)
            x_arg = x_col if x_col != y else None
            fig   = px.box(df, x=x_arg, y=y, color=color,
                           color_discrete_sequence=[single], **common)

        elif t == "Heatmap":
            if (x_col and y_col and y_col != "—"
                    and x_col in num_cols and y_col in num_cols
                    and x_col != y_col):
                cols_to_use = [x_col, y_col]
            else:
                cols_to_use = num_cols
            corr = df[cols_to_use].corr(method=corr_method).round(3)
            fig  = px.imshow(
                corr,
                color_continuous_scale=palette["scale"],
                title=title or f"מטריצת קורלציה ({corr_method})",
                text_auto=True,
            )

        else:
            fig = go.Figure()
            fig.add_annotation(text=f"סוג גרף לא מוכר: {chart_type}", showarrow=False)

        return apply_chart_style(fig)

    except Exception as exc:
        err_fig = go.Figure()
        err_fig.add_annotation(text=f"שגיאה ביצירת גרף: {exc}", showarrow=False,
                               font={"color": "#c5221f"})
        return apply_chart_style(err_fig)


# ─── Custom CSS ───────────────────────────────────────────────────────────────
def inject_css(lang: str) -> None:
    direction = "rtl" if lang == "he" else "ltr"
    text_align = "right" if lang == "he" else "left"
    st.markdown(f"""
<style>
/* ── General ── */
body {{ font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }}

/* ── RTL / LTR ── */
.stChatMessage p, .stChatMessage li {{
    direction: {direction};
    text-align: {text_align};
}}

/* ── Header ── */
.app-header {{
    padding: 1rem 0 0.5rem;
    border-bottom: 2px solid #f0f0f0;
    margin-bottom: 1rem;
}}
.app-title {{
    font-size: 2rem;
    font-weight: 700;
    color: #1a1a2e;
    direction: {direction};
}}
.app-sub {{
    color: #666;
    font-size: 0.95rem;
    direction: {direction};
}}

/* ── Sidebar sections ── */
.sidebar-section {{
    background: #f8f9fc;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.8rem;
    border: 1px solid #e8eaf0;
}}
.sidebar-label {{
    font-size: 0.78rem;
    font-weight: 600;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
    direction: {direction};
}}

/* ── Metric cards ── */
.metric-row {{
    display: flex;
    gap: 0.5rem;
    margin-top: 0.4rem;
}}
.metric-card {{
    flex: 1;
    background: white;
    border-radius: 8px;
    padding: 0.5rem 0.7rem;
    border: 1px solid #e0e4ef;
    text-align: center;
}}
.metric-value {{
    font-size: 1.2rem;
    font-weight: 700;
    color: #1a73e8;
}}
.metric-label {{
    font-size: 0.7rem;
    color: #888;
}}

/* ── Quick question buttons ── */
.stButton > button {{
    width: 100%;
    text-align: {text_align};
    direction: {direction};
    padding: 0.4rem 0.7rem;
    border-radius: 8px;
    font-size: 0.85rem;
    border: 1px solid #dde1f0;
    background: white;
    color: #333;
    transition: all 0.15s;
}}
.stButton > button:hover {{
    background: #1a73e8;
    color: white;
    border-color: #1a73e8;
}}

/* ── Chat area ── */
.stChatMessage {{
    border-radius: 12px;
    margin-bottom: 4px;
}}

/* ── Tool badge ── */
.tool-call-badge {{
    display: inline-block;
    background: #e8f0fe;
    color: #1a73e8;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 4px 0;
}}

/* ── Chart container ── */
.chart-wrap {{
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #e8eaf0;
    margin-top: 0.5rem;
}}

/* ── Welcome card ── */
.welcome-card {{
    background: linear-gradient(135deg, #667eea20, #764ba220);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border: 1px solid #dde1f0;
    direction: {direction};
}}

/* ── API key status badge ── */
.api-badge-ok {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #e6f4ea;
    color: #1e7e34;
    border: 1px solid #b7dfbe;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    width: 100%;
    justify-content: center;
    margin: 4px 0;
}}
.api-badge-err {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #fce8e6;
    color: #c5221f;
    border: 1px solid #f5c2c0;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    width: 100%;
    justify-content: center;
    margin: 4px 0;
}}
.api-hint {{
    font-size: 0.72rem;
    color: #888;
    text-align: center;
    margin-top: 4px;
}}
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ────────────────────────────────────────────────────────
def init_state() -> None:
    if "lang" not in st.session_state:
        st.session_state.lang = "he"
    if "messages" not in st.session_state:
        st.session_state.messages = []   # [{role, content, charts:[Path]}]
    if "chat" not in st.session_state:
        st.session_state.chat = None
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "pending_input" not in st.session_state:
        st.session_state.pending_input = None
    if "dashboard_charts" not in st.session_state:
        st.session_state.dashboard_charts = []
    if "data_warnings" not in st.session_state:
        st.session_state.data_warnings = []
    # API key: prefer env var, otherwise empty (user must supply via UI)
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if "api_key_from_env" not in st.session_state:
        st.session_state.api_key_from_env = bool(os.environ.get("ANTHROPIC_API_KEY", ""))


def get_active_key() -> str:
    """Return the API key currently in session (env var takes precedence)."""
    return st.session_state.api_key


def build_chat() -> ChatAnthropic:
    key = get_active_key()
    chat = ChatAnthropic(
        model="claude-opus-4-6",
        system_prompt=SYSTEM_PROMPT,
        max_tokens=8192,
        api_key=key if key else None,   # None → chatlas falls back to env var
    )
    chat.register_tool(tools.get_data_overview)
    chat.register_tool(tools.run_analysis)
    chat.register_tool(tools.create_chart)
    chat.register_tool(tools.suggest_next_analyses)
    return chat


# ─── Sample Data ──────────────────────────────────────────────────────────────
def make_sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    products  = ["Laptop","Phone","Tablet","Headphones","Monitor",
                 "Keyboard","Mouse","Webcam","Speaker","Charger"]
    regions   = ["North","South","East","West","Center"]
    category  = {"Laptop":"Computers","Phone":"Mobile","Tablet":"Mobile",
                 "Headphones":"Audio","Monitor":"Computers","Keyboard":"Accessories",
                 "Mouse":"Accessories","Webcam":"Accessories","Speaker":"Audio",
                 "Charger":"Accessories"}
    base_price= {"Laptop":1200,"Phone":800,"Tablet":600,"Headphones":150,
                 "Monitor":400,"Keyboard":80,"Mouse":50,"Webcam":90,
                 "Speaker":120,"Charger":30}

    dates = pd.date_range("2024-01-01","2024-12-31", periods=n)
    prod  = rng.choice(products, n)
    df = pd.DataFrame({
        "date":        dates,
        "product":     prod,
        "category":    [category[p] for p in prod],
        "region":      rng.choice(regions, n),
        "quantity":    rng.integers(1, 20, n),
        "unit_price":  [base_price[p] * rng.uniform(0.8, 1.2) for p in prod],
        "customer_age":rng.integers(18, 70, n),
        "satisfaction":rng.uniform(1, 5, n).round(1),
        "return_rate": rng.uniform(0, 0.15, n).round(3),
    })
    df["revenue"] = (df["quantity"] * df["unit_price"]).round(2)
    df["month"]   = df["date"].dt.month_name()
    # sprinkle a few NaNs
    df.loc[rng.choice(n, 15, replace=False), "satisfaction"] = np.nan
    return df


def load_uploaded_file(uploaded) -> pd.DataFrame:
    ext = Path(uploaded.name).suffix.lower()
    loaders = {
        ".csv":     lambda: pd.read_csv(uploaded),
        ".tsv":     lambda: pd.read_csv(uploaded, sep="\t"),
        ".xlsx":    lambda: pd.read_excel(uploaded),
        ".xls":     lambda: pd.read_excel(uploaded),
        ".json":    lambda: pd.read_json(uploaded),
        ".parquet": lambda: pd.read_parquet(uploaded),
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported file type: {ext}")
    return loaders[ext]()


# ─── Response Streaming Generator ─────────────────────────────────────────────
def text_stream(chat: ChatAnthropic, prompt: str):
    """Yield text chunks from chatlas stream (skip tool call/result objects)."""
    for chunk in chat.stream(prompt):
        if isinstance(chunk, str):
            yield chunk
        # ContentToolRequest / ContentToolResult are silently consumed here;
        # the tools run as a side-effect and populate tools._pending_charts


# ─── API Key Section ──────────────────────────────────────────────────────────
def render_api_key_section(T: dict) -> None:
    """Secure API key input — stored only in session_state (in-memory)."""
    st.markdown(f'<div class="sidebar-label">{T["api_header"]}</div>',
                unsafe_allow_html=True)

    has_key = bool(st.session_state.api_key)

    if has_key:
        # Show status badge
        label = T["api_from_env"] if st.session_state.api_key_from_env else T["api_set"]
        st.markdown(f'<div class="api-badge-ok">{label}</div>', unsafe_allow_html=True)

        # Allow clearing (only if not forced by env var)
        if not st.session_state.api_key_from_env:
            if st.button(T["api_clear"], use_container_width=True, key="api_clear_btn"):
                st.session_state.api_key = ""
                st.session_state.chat = None
                st.session_state.messages = []
                st.rerun()
    else:
        # Show error badge
        st.markdown(f'<div class="api-badge-err">{T["api_missing"]}</div>',
                    unsafe_allow_html=True)

        # Password input — masked, never echoed
        new_key = st.text_input(
            T["api_label"],
            type="password",
            placeholder=T["api_placeholder"],
            help=T["api_help"],
            label_visibility="collapsed",
            key="api_input_field",
        )
        if st.button(T["api_save"], use_container_width=True, key="api_save_btn"):
            cleaned = new_key.strip()
            if cleaned.startswith("sk-ant-"):
                st.session_state.api_key = cleaned
                st.session_state.api_key_from_env = False
                # Rebuild chat with new key if data is loaded
                if st.session_state.data_loaded:
                    st.session_state.chat = build_chat()
                st.rerun()
            else:
                st.error("המפתח חייב להתחיל ב־ sk-ant-  /  Key must start with sk-ant-")

        st.markdown(
            f'<div class="api-hint">🔗 {T["api_hint"]}</div>',
            unsafe_allow_html=True,
        )


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar(T: dict) -> None:
    with st.sidebar:
        # ── Language toggle ────────────────────────────────────────────────
        if st.button(T["lang_btn"], use_container_width=False, key="lang_toggle"):
            st.session_state.lang = "en" if st.session_state.lang == "he" else "he"
            st.rerun()

        st.markdown("---")

        # ── API Key ────────────────────────────────────────────────────────
        render_api_key_section(T)

        st.markdown("---")

        # ── File upload ────────────────────────────────────────────────────
        st.markdown(f'<div class="sidebar-label">{T["upload_header"]}</div>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader(
            T["upload_label"],
            type=["csv","tsv","xlsx","xls","json","parquet"],
            label_visibility="collapsed",
            help=T["upload_help"],
        )
        if uploaded:
            try:
                df = load_uploaded_file(uploaded)
                tools.set_dataframe(df, name=uploaded.name)
                st.session_state.data_loaded = True
                # Reset chat when new file is uploaded
                st.session_state.chat = build_chat()
                st.session_state.messages = []
                st.session_state.dashboard_charts = []
                st.session_state.data_warnings = validate_dataframe(df)
                st.success(f"✅ {uploaded.name}")
            except Exception as e:
                st.error(str(e))

        # ── Demo button ────────────────────────────────────────────────────
        if st.button(T["demo_btn"], use_container_width=True):
            df = make_sample_df()
            tools.set_dataframe(df, name="sample_sales.csv")
            st.session_state.data_loaded = True
            st.session_state.chat = build_chat()
            st.session_state.messages = []
            st.session_state.dashboard_charts = []
            st.session_state.data_warnings = validate_dataframe(df)
            st.success(T["demo_loaded"])

        st.markdown("---")

        # ── Dataset info ───────────────────────────────────────────────────
        df = tools.get_dataframe()
        if df is not None:
            st.markdown(f'<div class="sidebar-label">{T["data_header"]}</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="metric-value">{len(df):,}</div>
    <div class="metric-label">{T["rows"]}</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{len(df.columns)}</div>
    <div class="metric-label">{T["cols"]}</div>
  </div>
</div>""", unsafe_allow_html=True)

            # Column types summary
            dtype_counts = df.dtypes.astype(str).value_counts()
            type_str = " · ".join(f"{v}× {k}" for k, v in dtype_counts.items())
            st.caption(type_str)

            with st.expander("Columns"):
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    null_pct = df[col].isna().mean() * 100
                    null_str = f" ⚠️{null_pct:.0f}%" if null_pct > 0 else ""
                    st.caption(f"`{col}` — {dtype}{null_str}")

            st.markdown("---")

        else:
            st.info(T["no_data_warn"])

        # ── Quick questions ────────────────────────────────────────────────
        if df is not None:
            st.markdown(f'<div class="sidebar-label">{T["quick_header"]}</div>',
                        unsafe_allow_html=True)
            for q in T["quick_qs"]:
                if st.button(q, key=f"quick_{q}", use_container_width=True):
                    st.session_state.pending_input = q

        st.markdown("---")

        # ── Clear ──────────────────────────────────────────────────────────
        if st.button(T["clear_btn"], use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat = build_chat() if st.session_state.data_loaded else None
            st.rerun()


# ─── Chat History ─────────────────────────────────────────────────────────────
def render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Replay charts saved in this message
            for chart_path in msg.get("charts", []):
                p = Path(chart_path)
                if p.exists():
                    st.image(str(p), use_container_width=True)


# ─── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    init_state()
    lang = st.session_state.lang
    T = TEXT[lang]
    inject_css(lang)

    # ── Header ────────────────────────────────────────────────────────────
    col_title, col_badge = st.columns([5, 1])
    with col_title:
        st.markdown(f"""
<div class="app-header">
  <div class="app-title">{T["page_title"]}</div>
  <div class="app-sub">{T["page_sub"]}</div>
</div>""", unsafe_allow_html=True)
    with col_badge:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("claude-opus-4-6")

    # ── Sidebar ───────────────────────────────────────────────────────────
    render_sidebar(T)

    # ── Ensure chat object exists ─────────────────────────────────────────
    if st.session_state.chat is None and st.session_state.data_loaded:
        st.session_state.chat = build_chat()

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_chat, tab_charts, tab_dashboard, tab_data = st.tabs([
        T["tab_chat"], T["tab_charts"], T["tab_dashboard"], T["tab_data"],
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # Tab 1 – AI Chat
    # ═══════════════════════════════════════════════════════════════════════
    with tab_chat:
        if not st.session_state.messages:
            st.markdown(f"""
<div class="welcome-card">

{T["welcome"]}

</div>""", unsafe_allow_html=True)
        else:
            render_history()

        # Resolve quick-question button presses
        pre_fill   = st.session_state.pop("pending_input", None)
        no_key     = not bool(get_active_key())
        no_data    = tools.get_dataframe() is None
        user_input = st.chat_input(
            T["input_placeholder"],
            disabled=(no_key or no_data),
        )

        final_input = pre_fill or user_input

        if final_input and st.session_state.chat:
            with st.chat_message("user"):
                st.markdown(final_input)
            st.session_state.messages.append({
                "role": "user", "content": final_input, "charts": [],
            })

            with st.chat_message("assistant"):
                with st.spinner(T["thinking"]):
                    time.sleep(0.05)
                full_text = st.write_stream(
                    text_stream(st.session_state.chat, final_input)
                )
                chart_paths = tools.get_pending_charts()
                for chart_path in chart_paths:
                    if chart_path.exists():
                        st.image(str(chart_path), use_container_width=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_text or "",
                "charts": [str(p) for p in chart_paths],
            })

        elif final_input and not st.session_state.chat:
            st.warning(T["no_data_warn"])

    # ═══════════════════════════════════════════════════════════════════════
    # Tab 2 – Chart Builder
    # ═══════════════════════════════════════════════════════════════════════
    with tab_charts:
        df_now = tools.get_dataframe()
        if df_now is None:
            st.info(T["no_data_warn"])
        else:
            num_c = df_now.select_dtypes(include="number").columns.tolist()

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                chart_type_sel = st.selectbox(T["chart_type"], CHART_TYPES, key="cb_type")
            with col_b:
                x_col_sel = st.selectbox(T["chart_x"], df_now.columns.tolist(), key="cb_x")
            with col_c:
                y_col_sel = st.selectbox(T["chart_y"], ["—"] + num_c, key="cb_y")

            col_d, col_e, col_f = st.columns(3)
            with col_d:
                color_sel = st.selectbox(T["chart_color_by"], ["—"] + df_now.columns.tolist(), key="cb_color")
                chart_color = None if color_sel == "—" else color_sel
            with col_e:
                palette_sel = st.selectbox(T["chart_palette"], list(COLOR_PALETTES.keys()), key="cb_palette")
            with col_f:
                chart_title_inp = st.text_input(
                    T["chart_title_lbl"], placeholder=T["chart_title_ph"], key="cb_title"
                )

            # Correlation method — only for Heatmap
            corr_method_val = "pearson"
            if chart_type_sel == "Heatmap":
                corr_method_val = st.selectbox(
                    T["chart_corr_method"],
                    ["pearson", "spearman", "kendall"],
                    format_func=lambda x: {
                        "pearson": "פירסון (Pearson)",
                        "spearman": "ספירמן (Spearman)",
                        "kendall": "קנדל (Kendall)",
                    }[x],
                    key="cb_corr",
                )

            max_sample    = max(50, min(5_000, len(df_now)))
            chart_sample  = st.slider(
                T["chart_sample_lbl"],
                min_value=50, max_value=max_sample,
                value=min(500, max_sample), step=50,
                key="cb_sample",
            )

            cfg = {
                "type":        chart_type_sel,
                "x":           x_col_sel,
                "y":           y_col_sel,
                "color":       chart_color,
                "palette":     palette_sel,
                "title":       chart_title_inp,
                "corr_method": corr_method_val,
            }

            try:
                fig = build_chart(cfg, df_now, sample_size=chart_sample)
                st.plotly_chart(fig, use_container_width=True, key="chart_preview")
            except Exception as e:
                st.error(f"שגיאה ביצירת גרף: {e}")

            # Sample-size caption (not shown for chart types that use the full df)
            if chart_type_sel not in ("Heatmap", "עוגה / Pie", "היסטוגרמה / Histogram",
                                      "Box Plot"):
                st.caption(T["chart_sample_note"].format(n=chart_sample, total=len(df_now)))

            if st.button(T["add_to_dash"], key="add_to_dashboard"):
                cfg_to_save = {**cfg, "sample_size": chart_sample}
                st.session_state.dashboard_charts.append(cfg_to_save)
                st.success(f'{T["chart_added"]} '
                           f'סה"כ {len(st.session_state.dashboard_charts)} גרפים.')

    # ═══════════════════════════════════════════════════════════════════════
    # Tab 3 – Dashboard
    # ═══════════════════════════════════════════════════════════════════════
    with tab_dashboard:
        df_now = tools.get_dataframe()
        if df_now is None:
            st.info(T["no_data_warn"])
        elif not st.session_state.dashboard_charts:
            st.info(T["no_charts_hint"])
        else:
            hdr1, hdr2, hdr3 = st.columns([3, 1, 1])
            with hdr1:
                st.markdown(
                    f"**{len(st.session_state.dashboard_charts)} {T['dash_charts_count']}**"
                )
            with hdr2:
                grid_cols = st.radio(
                    T["dash_layout"], [1, 2, 3], horizontal=True, index=1,
                    format_func=lambda x: f"{x} {T['dash_col_suffix']}",
                    key="dash_grid",
                )
            with hdr3:
                if st.button(T["dash_clear"], use_container_width=True, key="dash_clear_btn"):
                    st.session_state.dashboard_charts = []
                    st.rerun()

            st.markdown("---")

            charts_list = st.session_state.dashboard_charts
            i = 0
            while i < len(charts_list):
                cols = st.columns(grid_cols)
                for j in range(grid_cols):
                    idx = i + j
                    if idx < len(charts_list):
                        c = charts_list[idx]
                        with cols[j]:
                            lbl = c.get("title") or f"{c.get('type','גרף')} – {c.get('x','')}"
                            st.markdown(f"**{lbl}**")
                            try:
                                fig = build_chart(c, df_now, sample_size=c.get("sample_size", 500))
                                st.plotly_chart(fig, use_container_width=True,
                                                key=f"dash_{idx}")
                            except Exception as e:
                                st.error(f"שגיאה: {e}")
                            if st.button(T["dash_remove"], key=f"rm_dash_{idx}",
                                         use_container_width=True):
                                st.session_state.dashboard_charts.pop(idx)
                                st.rerun()
                i += grid_cols

    # ═══════════════════════════════════════════════════════════════════════
    # Tab 4 – Data View
    # ═══════════════════════════════════════════════════════════════════════
    with tab_data:
        df_now = tools.get_dataframe()
        if df_now is None:
            st.info(T["no_data_warn"])
        else:
            st.markdown(f"### {T['data_title']}")

            # ── Data-quality warnings ──────────────────────────────────
            if st.session_state.data_warnings:
                with st.expander(
                    f"⚠️ {len(st.session_state.data_warnings)} {T['data_warnings_hdr']}",
                    expanded=True,
                ):
                    for w in st.session_state.data_warnings:
                        st.warning(w)
                    if st.button(T["data_autofix_btn"], key="auto_fix_data"):
                        fixed = auto_fix_dataframe(df_now)
                        tools.set_dataframe(fixed, name=tools.get_data_name())
                        st.session_state.data_warnings = []
                        st.success(T["data_fixed"])
                        st.rerun()

            # ── Search & filter ───────────────────────────────────────
            col1, col2 = st.columns(2)
            with col1:
                search = st.text_input(
                    T["data_search"], placeholder=T["data_search_ph"], key="data_search_input"
                )
            with col2:
                filter_col = st.selectbox(
                    T["data_filter_col"],
                    [T["data_filter_all"]] + df_now.columns.tolist(),
                    key="data_filter_col_sel",
                )

            display_df = df_now.copy()
            if search:
                if filter_col != T["data_filter_all"] and filter_col in display_df.columns:
                    mask = (
                        display_df[filter_col]
                        .astype(str)
                        .str.contains(search, case=False, na=False)
                    )
                else:
                    mask = display_df.astype(str).apply(
                        lambda col: col.str.contains(search, case=False, na=False)
                    ).any(axis=1)
                display_df = display_df[mask]

            st.dataframe(display_df, use_container_width=True, height=420)
            st.caption(f"מציג {len(display_df):,} מתוך {len(df_now):,} שורות")

            # ── Detailed stats ────────────────────────────────────────
            if st.checkbox(T["data_stats_chk"], key="data_stats_cb"):
                num_only = df_now.select_dtypes(include="number")
                if not num_only.empty:
                    st.dataframe(num_only.describe().round(2), use_container_width=True)
                else:
                    st.info("אין עמודות מספריות לסטטיסטיקות")


if __name__ == "__main__":
    main()
