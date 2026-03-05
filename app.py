"""
🤖 Data Analyst Chatbot — Streamlit App
Bilingual (Hebrew / English) interface powered by chatlas + Claude Opus 4.6
"""

import os
import sys
import logging
import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
from chatlas import ChatAnthropic

# ─── Structured Logging (SRE) ─────────────────────────────────────────────────
_LOG_FILE = Path(__file__).parent / "audit.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(_LOG_FILE), mode="a", encoding="utf-8"),
    ],
)
_logger = logging.getLogger("data_analyst_bot")

def _audit(action: str, detail: str = "") -> None:
    """Write a structured audit log entry."""
    _logger.info("ACTION=%s DETAIL=%s", action, str(detail)[:300])

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
    initial_sidebar_state="auto",   # collapsed on mobile, expanded on desktop
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
        # ספק AI
        "provider_header":    "🤖 ספק AI",
        "provider_cloud":     "☁️ Anthropic Claude",
        "provider_local":     "🦙 Ollama (מקומי)",
        "ollama_model_lbl":   "מודל Ollama",
        "ollama_status_ok":   "✅ Ollama פעיל",
        "ollama_status_err":  "❌ Ollama לא נמצא",
        "ollama_hint":        "הורד והפעל Ollama: ollama.com",
        "ollama_no_models":   "לא נמצאו מודלים. הפעל: ollama pull llama3.2",
        "ollama_refresh":     "🔄 רענן",
        # הודעות נוספות
        "no_key_hint":        "⚠️ הגדר מפתח API או חבר Ollama בסייד-בר",
        "columns_expander":   "עמודות",
        "data_showing":       "מציג {n:,} מתוך {total:,} שורות",
        "no_numeric_cols":    "אין עמודות מספריות לסטטיסטיקות",
        "chart_added_total":  "סה\"כ {n} גרפים בדשבורד",
        "chat_error":         "❌ שגיאה בעיבוד השאלה",
        "export_csv":         "⬇️ ייצא CSV",
        "export_csv_help":    "הורד את הטבלה המסוננת כקובץ CSV",
        # עיצוב גרפי AI
        "chart_settings_hdr": "🎨 עיצוב גרפי AI",
        "chart_seaborn_style":"סגנון רקע",
        "chart_ai_palette":   "פלטת צבעים",
        "chart_figsize":      "גודל גרף",
        # ייצוא
        "export_chat":           "📄 ייצא שיחה (HTML)",
        "export_chat_help":      "הורד את השיחה עם הגרפים כקובץ HTML",
        "export_dashboard":      "💾 ייצא דשבורד (HTML)",
        "export_dashboard_help": "הורד את כל הגרפים בדשבורד כ-HTML אינטראקטיבי",
        "export_ai_charts":      "🖼 הורד גרפי AI (ZIP)",
        "export_ai_charts_help": "הורד את כל גרפי ה-AI שנוצרו בשיחה",
        "export_pdf":            "📑 ייצא שיחה (PDF)",
        "export_pdf_help":       "הורד את השיחה כקובץ PDF עם גרפים",
        # קבצים גדולים / אבטחה / onboarding
        "large_file_notice":     "ℹ️ הקובץ גדול — מוצג מדגם של {n:,} שורות מתוך {total:,}",
        "rate_limit_warn":       "⚠️ {n}/{limit} שאלות בסשן זה",
        "rate_limit_block":      "🚫 הגעת למגבלת {limit} שאלות בסשן. רענן את הדף להמשך.",
        "input_too_long":        "⚠️ השאלה קוצרה ל-{max} תווים",
        "onboarding_hint":       "💡 **איך להתחיל?** העלה קובץ ← שאל שאלה ב-AI Chat ← בנה גרפים ← הוסף לדשבורד",
        "memory_warn":           "⚠️ הנתונים תופסים {mb} MB — ייתכן שהאפליקציה תהיה איטית",
        # Technical Writer
        "whats_new_hdr":         "🆕 מה חדש",
        # Accessibility
        "a11y_skip":             "דלג לתוכן הראשי",
        # Ethics
        "bias_warning":          "⚠️ עמודות רגישות זוהו: **{cols}** — שים לב להטיות אפשריות בניתוח",
        "privacy_notice":        "🔒 **הודעת פרטיות:** הנתונים שלך נשלחים ל-API חיצוני (Anthropic) לצורך הניתוח. אל תעלה נתונים אישיים מזהים ללא הרשאה מתאימה.",
        "privacy_ok":            "הבנתי ✓",
        # Email
        "email_hdr":             "📧 שלח במייל",
        "email_from":            "כתובת שולח (Gmail)",
        "email_password":        "App Password",
        "email_to":              "כתובת נמען",
        "email_subject":         "נושא",
        "email_what":            "מה לשלוח?",
        "email_what_html":       "שיחה (HTML)",
        "email_what_csv":        "נתונים (CSV)",
        "email_send":            "📤 שלח",
        "email_sent_ok":         "✅ המייל נשלח!",
        "email_sent_err":        "❌ שגיאה בשליחה: {err}",
        "email_hint":            "Gmail: הגדר App Password ב-myaccount.google.com/apppasswords",
        "email_no_data":         "⚠️ אין נתונים לשליחה",
        "email_no_chat":         "⚠️ אין שיחה לשליחה",
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
        # AI Provider
        "provider_header":    "🤖 AI Provider",
        "provider_cloud":     "☁️ Anthropic Claude",
        "provider_local":     "🦙 Ollama (local)",
        "ollama_model_lbl":   "Ollama Model",
        "ollama_status_ok":   "✅ Ollama running",
        "ollama_status_err":  "❌ Ollama not found",
        "ollama_hint":        "Download & run Ollama: ollama.com",
        "ollama_no_models":   "No models found. Run: ollama pull llama3.2",
        "ollama_refresh":     "🔄 Refresh",
        # Extra strings
        "no_key_hint":        "⚠️ Configure API key or connect Ollama in the sidebar",
        "columns_expander":   "Columns",
        "data_showing":       "Showing {n:,} of {total:,} rows",
        "no_numeric_cols":    "No numeric columns for statistics",
        "chart_added_total":  "{n} charts in dashboard",
        "chat_error":         "❌ Error processing your question",
        "export_csv":         "⬇️ Export CSV",
        "export_csv_help":    "Download the filtered table as a CSV file",
        # AI chart styling
        "chart_settings_hdr": "🎨 AI Chart Style",
        "chart_seaborn_style":"Background style",
        "chart_ai_palette":   "Color palette",
        "chart_figsize":      "Figure size",
        # Export
        "export_chat":           "📄 Export Conversation (HTML)",
        "export_chat_help":      "Download the conversation with charts as an HTML file",
        "export_dashboard":      "💾 Export Dashboard (HTML)",
        "export_dashboard_help": "Download all dashboard charts as an interactive HTML file",
        "export_ai_charts":      "🖼 Download AI Charts (ZIP)",
        "export_ai_charts_help": "Download all AI-generated charts from this conversation",
        "export_pdf":            "📑 Export Chat (PDF)",
        "export_pdf_help":       "Download the conversation as a PDF with charts",
        # Large files / security / onboarding
        "large_file_notice":     "ℹ️ Large file — showing a sample of {n:,} of {total:,} rows",
        "rate_limit_warn":       "⚠️ {n}/{limit} questions used this session",
        "rate_limit_block":      "🚫 You have reached the {limit}-question session limit. Refresh to continue.",
        "input_too_long":        "⚠️ Question trimmed to {max} characters",
        "onboarding_hint":       "💡 **Getting started:** Upload a file ← Ask in AI Chat ← Build Charts ← Add to Dashboard",
        "memory_warn":           "⚠️ Dataset is {mb} MB in memory — the app may run slowly",
        # Technical Writer
        "whats_new_hdr":         "🆕 What's New",
        # Accessibility
        "a11y_skip":             "Skip to main content",
        # Ethics
        "bias_warning":          "⚠️ Sensitive columns detected: **{cols}** — be aware of potential bias in analysis",
        "privacy_notice":        "🔒 **Privacy Notice:** Your data is sent to an external API (Anthropic) for analysis. Do not upload personally identifiable information without appropriate authorisation.",
        "privacy_ok":            "Got it ✓",
        # Email
        "email_hdr":             "📧 Send by Email",
        "email_from":            "From address (Gmail)",
        "email_password":        "App Password",
        "email_to":              "Recipient address",
        "email_subject":         "Subject",
        "email_what":            "What to send?",
        "email_what_html":       "Conversation (HTML)",
        "email_what_csv":        "Data (CSV)",
        "email_send":            "📤 Send",
        "email_sent_ok":         "✅ Email sent!",
        "email_sent_err":        "❌ Send error: {err}",
        "email_hint":            "Gmail: create an App Password at myaccount.google.com/apppasswords",
        "email_no_data":         "⚠️ No data to send",
        "email_no_chat":         "⚠️ No conversation to send",
    },
}

# PROMPT_VERSION = "2.0"  — bump this when the prompt changes significantly
SYSTEM_PROMPT = """
You are an expert, bilingual Data Analyst assistant.
Respond in the same language the user writes in (Hebrew or English).

## Analytical Process (Chain-of-Thought — always follow this order)
1. FIRST call get_data_overview() to understand the dataset structure and column names.
2. THEN call run_analysis() to compute the exact numbers needed to answer the question.
3. ONLY THEN write your response — using only the numbers you actually computed.
4. Visualize every key finding with create_chart().

## Guardrails (STRICT — never violate)
- ONLY report numbers and statistics that you computed with run_analysis().
  Never estimate, approximate, or invent figures.
- If you are unsure about a number, say "I need to verify this" and call run_analysis().
- NEVER suggest modifying, deleting, or writing back to the user's data.
- NEVER access external URLs, files, APIs, or services.
- NEVER execute operating-system commands or import new libraries.

## Response Format
1. 📊 **Key Findings** — bullet points with the exact numbers from run_analysis()
2. 🔍 **Interpretation** — what the numbers mean in context
3. 💡 **Insight / Recommendation** — business or analytical conclusion
4. ➡️ **Follow-up** — 1–2 natural follow-up questions

## Few-Shot Example
User: "What are the top 3 products by revenue?"
Thought process:
  → call get_data_overview() to confirm column names
  → call run_analysis("df.groupby('product')['revenue'].sum().nlargest(3)")
  → call create_chart(chart_type='barh', x_column='product', y_column='revenue', title='Top 3 Products by Revenue')
Response:
  📊 **Key Findings**
  - Laptop: $45,230 (32% of total)
  - Phone: $38,100 (27%)
  - Tablet: $21,500 (15%)
  🔍 **Interpretation** — Electronics dominate revenue; Laptop alone accounts for nearly a third.
  💡 **Insight** — Consider bundling Laptop with accessories to increase average basket size.
  ➡️ **Follow-up** — Want to see revenue trend by month for these products?

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

# ─── Ollama Helper ────────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def get_ollama_models() -> tuple:
    """Query local Ollama for available models. Returns (available: bool, model_names: list)."""
    try:
        import urllib.request as _req
        import json as _json
        req = _req.Request(
            "http://localhost:11434/api/tags",
            headers={"Accept": "application/json"},
        )
        with _req.urlopen(req, timeout=2) as resp:
            data = _json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        return True, models
    except Exception:
        return False, []


# ─── Data Validation ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def validate_dataframe(df: pd.DataFrame) -> list:
    """Return list of data-quality warnings (numeric-as-string, unparsed dates)."""
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


# ─── Export Helpers ───────────────────────────────────────────────────────────

def export_chat_html(messages: list, title: str = "Chat Export") -> bytes:
    """Return UTF-8 encoded HTML of the conversation with base64-embedded chart images."""
    import base64
    rows = []
    for msg in messages:
        is_user = msg["role"] == "user"
        bg = "#e8f0fe" if is_user else "#f1f8e9"
        border = "#1a73e8" if is_user else "#2ecc71"
        label = "👤 User" if is_user else "🤖 Assistant"
        content = (
            msg["content"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        rows.append(
            f'<div style="background:{bg};border-left:4px solid {border};'
            f'padding:12px 16px;border-radius:8px;margin:10px 0;">'
            f'<strong style="color:{border}">{label}</strong><br><br>{content}'
        )
        for chart_path in msg.get("charts", []):
            p = Path(chart_path)
            if p.exists():
                b64 = base64.b64encode(p.read_bytes()).decode()
                rows.append(
                    f'<br><img src="data:image/png;base64,{b64}" '
                    f'style="max-width:100%;border-radius:8px;margin-top:8px;">'
                )
        rows.append("</div>")
    body = "\n".join(rows) if rows else "<p>No messages.</p>"
    html = (
        "<!DOCTYPE html><html><head>"
        '<meta charset="utf-8">'
        f"<title>{title}</title>"
        "<style>"
        'body{font-family:"Segoe UI",Arial,sans-serif;max-width:860px;margin:40px auto;padding:0 20px;color:#1a1a2e;}'
        "h1{color:#1a73e8;border-bottom:2px solid #e8eaf0;padding-bottom:8px;}"
        "</style></head><body>"
        f"<h1>💬 {title}</h1>"
        f"{body}"
        "</body></html>"
    )
    return html.encode("utf-8")


def export_dashboard_html(charts_configs: list, df: pd.DataFrame,
                          title: str = "Dashboard") -> str:
    """Return a standalone HTML string with all dashboard charts (plotly CDN)."""
    import plotly.io as pio
    figs_html = []
    for cfg in charts_configs:
        try:
            fig = build_chart(cfg, df, sample_size=cfg.get("sample_size", 500))
            lbl = cfg.get("title") or f"{cfg.get('type', 'Chart')} – {cfg.get('x', '')}"
            fig_html = pio.to_html(fig, full_html=False, include_plotlyjs=False)
            figs_html.append(
                f'<div style="margin-bottom:2rem">'
                f'<h3 style="color:#444;border-bottom:1px solid #eee;padding-bottom:4px">{lbl}</h3>'
                f'{fig_html}</div>'
            )
        except Exception as exc:
            figs_html.append(f'<p style="color:red">Error: {exc}</p>')
    return (
        "<!DOCTYPE html><html><head>"
        '<meta charset="utf-8">'
        f"<title>{title}</title>"
        '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
        "<style>"
        'body{font-family:"Segoe UI",Arial,sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;}'
        "h1{color:#1a73e8;}"
        "</style></head><body>"
        f"<h1>📈 {title}</h1>"
        + "".join(figs_html)
        + "</body></html>"
    )


def export_ai_charts_zip(messages: list) -> bytes:
    """Return a ZIP archive containing all AI-generated chart PNGs from the conversation."""
    import zipfile
    import io as _io
    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        seen: set = set()
        for msg in messages:
            for chart_path in msg.get("charts", []):
                p = Path(chart_path)
                if p.exists() and str(p) not in seen:
                    zf.write(p, p.name)
                    seen.add(str(p))
    return buf.getvalue()


def export_chat_pdf(messages: list, title: str = "Chat Export") -> bytes:
    """Return PDF bytes of the conversation with embedded chart images."""
    try:
        from fpdf import FPDF
        import matplotlib
        from bidi.algorithm import get_display
    except ImportError:
        return b""

    # DejaVuSans from matplotlib — full Unicode + Hebrew support
    font_path = (
        Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf"
    )

    def _is_rtl(text: str) -> bool:
        return any("\u0590" <= c <= "\u05FF" for c in text)

    def _fix(text: str) -> str:
        return get_display(text) if _is_rtl(text) else text

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    if font_path.exists():
        pdf.add_font("DejaVu", "", str(font_path), uni=True)
        fnt = "DejaVu"
    else:
        fnt = "Helvetica"

    pdf.add_page()
    pdf.set_font(fnt, size=18)
    pdf.cell(0, 10, _fix(title), align="R" if _is_rtl(title) else "L", ln=True)
    pdf.ln(4)

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        charts = msg.get("charts", [])

        # Role label
        pdf.set_font(fnt, size=9)
        pdf.set_fill_color(230, 240, 255) if role == "assistant" else pdf.set_fill_color(230, 250, 230)
        pdf.set_text_color(80, 80, 80)
        label = "Assistant:" if role == "assistant" else "User:"
        pdf.cell(0, 6, label, fill=True, ln=True)

        # Message lines
        pdf.set_font(fnt, size=10)
        pdf.set_text_color(30, 30, 30)
        for line in content.split("\n"):
            clean = line.strip()
            if not clean:
                pdf.ln(2)
                continue
            # Strip markdown bullets/headers to avoid junk chars
            clean = clean.lstrip("*#>-").strip()
            if not clean:
                continue
            fixed = _fix(clean)
            pdf.multi_cell(0, 5, fixed, align="R" if _is_rtl(clean) else "L")

        # Embedded chart images
        for chart_path in charts:
            p = Path(chart_path)
            if p.exists():
                if pdf.get_y() > 220:
                    pdf.add_page()
                pdf.image(str(p), x=10, w=180)
                pdf.ln(3)

        pdf.ln(3)

    return bytes(pdf.output())


def send_email_smtp(
    from_addr: str,
    password: str,
    to_addr: str,
    subject: str,
    body_html=None,
    attachment_bytes=None,
    attachment_name: str = "data.csv",
) -> None:
    """Send email via Gmail SMTP (TLS). Raises on failure."""
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    msg = MIMEMultipart("mixed")
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject

    msg.attach(MIMEText(body_html or subject, "html" if body_html else "plain", "utf-8"))

    if attachment_bytes:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment_bytes)
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f'attachment; filename="{attachment_name}"',
        )
        msg.attach(part)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.ehlo()
        server.starttls()
        server.login(from_addr, password)
        server.sendmail(from_addr, to_addr, msg.as_string())


# ─── Custom CSS ───────────────────────────────────────────────────────────────
def inject_css(lang: str) -> None:
    direction = "rtl" if lang == "he" else "ltr"
    text_align = "right" if lang == "he" else "left"
    st.markdown(f"""
<style>
/* ═══════════════════════════════════════════════
   BASE STYLES  (mobile-first)
═══════════════════════════════════════════════ */
*, *::before, *::after {{ box-sizing: border-box; }}

body {{
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    -webkit-text-size-adjust: 100%;   /* prevent iOS font inflation */
}}

/* ── RTL / LTR ── */
.stChatMessage p, .stChatMessage li {{
    direction: {direction};
    text-align: {text_align};
}}

/* ── Header ── */
.app-header {{
    padding: 0.75rem 0 0.5rem;
    border-bottom: 2px solid #f0f0f0;
    margin-bottom: 0.75rem;
}}
.app-title {{
    font-size: 1.5rem;        /* mobile default */
    font-weight: 700;
    color: #1a1a2e;
    direction: {direction};
    line-height: 1.2;
}}
.app-sub {{
    color: #666;
    font-size: 0.85rem;
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
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.4rem;
}}
.metric-card {{
    flex: 1;
    min-width: 70px;
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

/* ── All buttons — touch-friendly min size ── */
.stButton > button,
.stDownloadButton > button {{
    min-height: 44px;           /* Apple/Google touch target guideline */
    width: 100%;
    text-align: {text_align};
    direction: {direction};
    padding: 0.5rem 0.8rem;
    border-radius: 8px;
    font-size: 0.88rem;
    border: 1px solid #dde1f0;
    background: white;
    color: #333;
    transition: background 0.15s, color 0.15s, border-color 0.15s;
    cursor: pointer;
    -webkit-tap-highlight-color: transparent;
}}
.stButton > button:hover,
.stButton > button:focus-visible {{
    background: #1a73e8;
    color: white;
    border-color: #1a73e8;
    outline: none;
}}

/* ── Download buttons — distinct accent ── */
.stDownloadButton > button {{
    background: #f0f4ff;
    color: #1a73e8;
    border-color: #c5d6fa;
    font-weight: 600;
}}
.stDownloadButton > button:hover,
.stDownloadButton > button:focus-visible {{
    background: #1a73e8;
    color: white;
    border-color: #1a73e8;
    outline: none;
}}
.stDownloadButton > button[disabled] {{
    opacity: 0.45;
    pointer-events: none;
}}

/* ── Chat area ── */
.stChatMessage {{
    border-radius: 12px;
    margin-bottom: 4px;
}}
.stChatInput textarea {{
    font-size: 1rem;        /* legible on mobile */
    min-height: 48px;
    direction: {direction};
    text-align: {text_align};
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
    max-width: 100%;
}}

/* ── Welcome card ── */
.welcome-card {{
    background: linear-gradient(135deg, #667eea20, #764ba220);
    border-radius: 12px;
    padding: 1rem 1.2rem;
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
    padding: 6px 12px;
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
    padding: 6px 12px;
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

/* ═══════════════════════════════════════════════
   TABLET  ≥ 768 px
═══════════════════════════════════════════════ */
@media (min-width: 768px) {{
    .app-title {{
        font-size: 1.8rem;
    }}
    .app-sub {{
        font-size: 0.9rem;
    }}
    .stButton > button,
    .stDownloadButton > button {{
        font-size: 0.9rem;
        padding: 0.5rem 0.9rem;
    }}
}}

/* ═══════════════════════════════════════════════
   DESKTOP  ≥ 1024 px
═══════════════════════════════════════════════ */
@media (min-width: 1024px) {{
    .app-title {{
        font-size: 2rem;
    }}
    .app-sub {{
        font-size: 0.95rem;
    }}
    .stButton > button,
    .stDownloadButton > button {{
        font-size: 0.92rem;
    }}
    .welcome-card {{
        padding: 1.2rem 1.5rem;
    }}
}}

/* ═══════════════════════════════════════════════
   SMALL MOBILE  < 480 px
═══════════════════════════════════════════════ */
@media (max-width: 480px) {{
    .app-title {{
        font-size: 1.3rem;
    }}
    .metric-row {{
        flex-direction: column;
    }}
    .metric-card {{
        min-width: unset;
    }}
    .stButton > button,
    .stDownloadButton > button {{
        min-height: 48px;       /* extra large on phones */
        font-size: 0.95rem;
    }}
    /* Reduce horizontal whitespace */
    .block-container {{
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }}
}}

/* ═══════════════════════════════════════════════
   ACCESSIBILITY (WCAG 2.1 AA)
═══════════════════════════════════════════════ */

/* Skip-to-main-content link (keyboard users) */
.skip-link {{
    position: absolute;
    top: -40px;
    left: 0;
    background: #1a73e8;
    color: white;
    padding: 8px 16px;
    border-radius: 0 0 4px 0;
    font-weight: 600;
    z-index: 9999;
    text-decoration: none;
    transition: top 0.2s;
}}
.skip-link:focus {{
    top: 0;
}}

/* Visible focus ring on every interactive element */
*:focus-visible {{
    outline: 3px solid #1a73e8 !important;
    outline-offset: 2px !important;
    border-radius: 4px !important;
}}
/* Remove default outline only when :focus-visible is shown */
*:focus:not(:focus-visible) {{
    outline: none;
}}

/* Chart images — accessible alt description */
.chart-wrap img {{
    display: block;
    max-width: 100%;
}}

/* Ensure sidebar caption text meets 4.5:1 contrast */
.sidebar-label {{
    color: #595959;  /* 7:1 ratio on white */
}}
.metric-label {{
    color: #595959;
}}
.api-hint {{
    color: #595959;
}}

/* ═══════════════════════════════════════════════
   HIGH-DPI / RETINA (image sharpness)
═══════════════════════════════════════════════ */
@media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi) {{
    .chart-wrap img {{
        image-rendering: -webkit-optimize-contrast;
    }}
}}

/* ═══════════════════════════════════════════════
   DARK MODE  (respects OS preference)
═══════════════════════════════════════════════ */
@media (prefers-color-scheme: dark) {{
    .app-title {{ color: #e8eaf6; }}
    .app-sub   {{ color: #9fa8bc; }}
    .app-header {{ border-bottom-color: #2d2d44; }}

    .sidebar-label {{ color: #9fa8bc; }}

    .metric-card {{
        background: #1e1e2e;
        border-color: #2d2d44;
    }}
    .metric-value {{ color: #82aaff; }}
    .metric-label {{ color: #9fa8bc; }}

    .welcome-card {{
        background: linear-gradient(135deg, #667eea12, #764ba212);
        border-color: #2d2d44;
    }}

    .stButton > button {{
        background: #1e1e2e;
        color: #c9d1d9;
        border-color: #30363d;
    }}
    .stButton > button:hover {{
        background: #388bfd;
        color: white;
        border-color: #388bfd;
    }}
    .stDownloadButton > button {{
        background: #1c2d4a;
        color: #79b8ff;
        border-color: #1f4168;
    }}
    .stDownloadButton > button:hover {{
        background: #388bfd;
        color: white;
        border-color: #388bfd;
    }}

    .tool-call-badge {{
        background: #1c2d4a;
        color: #79b8ff;
    }}
    .chart-wrap {{ border-color: #2d2d44; }}

    .api-badge-ok {{
        background: #1a3a2a;
        color: #56d364;
        border-color: #2ea043;
    }}
    .api-badge-err {{
        background: #3a1a1a;
        color: #f85149;
        border-color: #da3633;
    }}
    .api-hint {{ color: #8b949e; }}
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
    # API key: env var → st.secrets → empty (user must supply via UI)
    if "api_key" not in st.session_state:
        env_key = os.environ.get("ANTHROPIC_API_KEY", "")
        try:
            secrets_key = st.secrets.get("ANTHROPIC_API_KEY", "") if not env_key else ""
        except Exception:
            secrets_key = ""
        st.session_state.api_key = env_key or secrets_key
        st.session_state.api_key_from_env = bool(env_key or secrets_key)
    if "api_key_from_env" not in st.session_state:
        st.session_state.api_key_from_env = False
    # Rate limiting: max N AI calls per session
    if "_request_count" not in st.session_state:
        st.session_state._request_count = 0
    # Provider: "anthropic" | "ollama"
    if "provider" not in st.session_state:
        st.session_state.provider = "anthropic"
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = ""
    if "ollama_available" not in st.session_state:
        st.session_state.ollama_available = False
    if "ollama_models" not in st.session_state:
        st.session_state.ollama_models = []
    if "ollama_queried" not in st.session_state:
        st.session_state.ollama_queried = False
    # Chart style (for AI/matplotlib charts)
    if "chart_seaborn_style" not in st.session_state:
        st.session_state.chart_seaborn_style = "whitegrid"
    if "chart_ai_palette" not in st.session_state:
        st.session_state.chart_ai_palette = "husl"
    if "chart_figsize_lbl" not in st.session_state:
        st.session_state.chart_figsize_lbl = "10×5"
    # Email config (cleared on browser close — stored only in session state)
    if "email_from" not in st.session_state:
        st.session_state.email_from = ""
    if "email_to" not in st.session_state:
        st.session_state.email_to = ""
    if "email_subject" not in st.session_state:
        st.session_state.email_subject = ""
    if "email_what" not in st.session_state:
        st.session_state.email_what = "html"
    # Ethics / Privacy
    if "_privacy_accepted" not in st.session_state:
        st.session_state._privacy_accepted = False


def get_active_key() -> str:
    """Return the API key currently in session (env var takes precedence)."""
    return st.session_state.api_key


def build_chat():
    """Build a chat object based on the current provider (Anthropic or Ollama)."""
    if st.session_state.get("provider", "anthropic") == "ollama":
        model_name = st.session_state.get("ollama_model", "")
        if not model_name:
            raise ValueError(
                "לא נבחר מודל Ollama — בחר מודל מהתפריט בסייד-בר לאחר הרצת: ollama pull llama3.2\n"
                "No Ollama model selected — run: ollama pull llama3.2, then choose from the sidebar."
            )
        from chatlas import ChatOllama
        chat = ChatOllama(
            model=model_name,
            system_prompt=SYSTEM_PROMPT,
        )
    else:
        key = get_active_key()
        chat = ChatAnthropic(
            model="claude-opus-4-6",
            system_prompt=SYSTEM_PROMPT,
            max_tokens=8192,
            api_key=key if key else None,
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


_SENSITIVE_COLS = {
    # English
    "gender", "sex", "race", "ethnicity", "religion", "nationality",
    "age", "disability", "sexual_orientation", "political_affiliation",
    "marital_status", "salary", "income", "wage", "creditScore", "credit_score",
    # Hebrew
    "מגדר", "גזע", "דת", "לאום", "גיל", "נכות", "משכורת", "הכנסה",
}

_MEMORY_WARN_MB = 400  # warn when in-memory DataFrame exceeds this size


def check_df_memory(df: pd.DataFrame) -> float:
    """Return DataFrame memory usage in MB."""
    return df.memory_usage(deep=True).sum() / 1024 / 1024


def detect_sensitive_columns(df: pd.DataFrame) -> list:
    """Return column names that may contain sensitive / protected attributes."""
    found = []
    for col in df.columns:
        col_lower = col.lower().replace(" ", "_").replace("-", "_")
        if any(s in col_lower for s in _SENSITIVE_COLS):
            found.append(col)
    return found


def _friendly_error(err: Exception) -> str:
    """Map common API / runtime exceptions to bilingual human-readable messages."""
    msg = str(err).lower()
    if "rate_limit" in msg or "429" in msg:
        return "⏳ שרת ה-AI עמוס — נסה שוב בעוד כמה שניות / AI server busy — retry in a moment"
    if "authentication" in msg or "401" in msg or ("invalid" in msg and "key" in msg):
        return "🔑 מפתח ה-API שגוי — בדוק בסייד-בר / Invalid API key — check the sidebar"
    if "timeout" in msg or "timed out" in msg:
        return "⏱️ פסק הזמן — הנתונים עשויים להיות גדולים / Request timed out — data may be too large"
    if "connection" in msg or "network" in msg:
        return "🌐 בעיית חיבור לרשת / Network connection error"
    if "context_length" in msg or "too long" in msg:
        return "📏 הנתונים ארוכים מדי עבור הבקשה — נסה לסנן את הטבלה / Data too long — try filtering first"
    return f"❌ {str(err)[:250]}"


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
def text_stream(chat, prompt: str):
    """Yield text chunks from chatlas stream (skip tool call/result objects)."""
    for chunk in chat.stream(prompt):
        if isinstance(chunk, str):
            yield chunk
        # ContentToolRequest / ContentToolResult are silently consumed here;
        # the tools run as a side-effect and populate tools._pending_charts


# ─── Provider Section ─────────────────────────────────────────────────────────

def _render_anthropic_key(T: dict) -> None:
    """Anthropic API key sub-section (stored in-memory only)."""
    has_key = bool(st.session_state.api_key)
    if has_key:
        label = T["api_from_env"] if st.session_state.api_key_from_env else T["api_set"]
        st.markdown(f'<div class="api-badge-ok">{label}</div>', unsafe_allow_html=True)
        if not st.session_state.api_key_from_env:
            if st.button(T["api_clear"], use_container_width=True, key="api_clear_btn"):
                st.session_state.api_key = ""
                st.session_state.chat = None
                st.session_state.messages = []
                st.rerun()
    else:
        st.markdown(f'<div class="api-badge-err">{T["api_missing"]}</div>',
                    unsafe_allow_html=True)
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
                if st.session_state.data_loaded:
                    st.session_state.chat = build_chat()
                st.rerun()
            else:
                st.error("המפתח חייב להתחיל ב־ sk-ant-  /  Key must start with sk-ant-")
        st.markdown(f'<div class="api-hint">🔗 {T["api_hint"]}</div>',
                    unsafe_allow_html=True)


def _render_ollama_section(T: dict) -> None:
    """Ollama local-model sub-section."""
    # Auto-query once per session
    if not st.session_state.ollama_queried:
        available, models = get_ollama_models()
        st.session_state.ollama_available = available
        st.session_state.ollama_models = models
        st.session_state.ollama_queried = True
        if models and not st.session_state.ollama_model:
            st.session_state.ollama_model = models[0]

    col_status, col_refresh = st.columns([3, 1])
    with col_status:
        if st.session_state.ollama_available:
            st.markdown(f'<div class="api-badge-ok">{T["ollama_status_ok"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="api-badge-err">{T["ollama_status_err"]}</div>',
                        unsafe_allow_html=True)
    with col_refresh:
        if st.button(T["ollama_refresh"], key="ollama_refresh_btn", use_container_width=True):
            available, models = get_ollama_models()
            st.session_state.ollama_available = available
            st.session_state.ollama_models = models
            if models and not st.session_state.ollama_model:
                st.session_state.ollama_model = models[0]
            if st.session_state.data_loaded:
                st.session_state.chat = build_chat()
            st.rerun()

    if st.session_state.ollama_available and st.session_state.ollama_models:
        models_list = st.session_state.ollama_models
        cur = st.session_state.ollama_model
        idx = models_list.index(cur) if cur in models_list else 0
        model_choice = st.selectbox(
            T["ollama_model_lbl"],
            models_list,
            index=idx,
            key="ollama_model_select",
            label_visibility="collapsed",
        )
        if model_choice != st.session_state.ollama_model:
            st.session_state.ollama_model = model_choice
            st.session_state.chat = build_chat() if st.session_state.data_loaded else None
    elif st.session_state.ollama_available:
        st.caption(T["ollama_no_models"])
    else:
        st.markdown(f'<div class="api-hint">🔗 {T["ollama_hint"]}</div>',
                    unsafe_allow_html=True)


def render_provider_section(T: dict) -> None:
    """Provider toggle: Anthropic Cloud ↔ Ollama Local."""
    st.markdown(f'<div class="sidebar-label">{T["provider_header"]}</div>',
                unsafe_allow_html=True)
    cur_provider = st.session_state.get("provider", "anthropic")
    provider = st.radio(
        T["provider_header"],
        options=["anthropic", "ollama"],
        format_func=lambda x: T["provider_cloud"] if x == "anthropic" else T["provider_local"],
        index=0 if cur_provider == "anthropic" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="provider_radio",
    )
    if provider != cur_provider:
        st.session_state.provider = provider
        st.session_state.chat = None
        st.rerun()

    st.markdown('<div style="margin-top:0.4rem"></div>', unsafe_allow_html=True)
    if st.session_state.provider == "ollama":
        _render_ollama_section(T)
    else:
        _render_anthropic_key(T)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar(T: dict) -> None:
    with st.sidebar:
        # ── Language toggle ────────────────────────────────────────────────
        if st.button(T["lang_btn"], use_container_width=False, key="lang_toggle"):
            st.session_state.lang = "en" if st.session_state.lang == "he" else "he"
            st.rerun()

        st.markdown("---")

        # ── AI Provider ────────────────────────────────────────────────────
        render_provider_section(T)

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
        _MAX_ROWS = 100_000
        if uploaded:
            file_id = f"{uploaded.name}_{uploaded.size}"
            if file_id != st.session_state.get("_uploaded_file_id"):
                try:
                    df = load_uploaded_file(uploaded)
                    _total_rows = len(df)
                    # ── Large file sampling ────────────────────────────────
                    if _total_rows > _MAX_ROWS:
                        df = df.sample(_MAX_ROWS, random_state=42).reset_index(drop=True)
                        st.info(T["large_file_notice"].format(n=_MAX_ROWS, total=_total_rows))
                    # ── Memory check (SRE) ────────────────────────────────
                    _mb = check_df_memory(df)
                    if _mb > _MEMORY_WARN_MB:
                        st.warning(T["memory_warn"].format(mb=int(_mb)))
                    # ── Bias / sensitive-column detection (Ethics) ────────
                    _sensitive = detect_sensitive_columns(df)
                    if _sensitive:
                        st.warning(T["bias_warning"].format(cols=", ".join(_sensitive)))
                    tools.set_dataframe(df, name=uploaded.name)
                    st.session_state.data_loaded = True
                    st.session_state._uploaded_file_id = file_id
                    st.session_state.chat = None
                    st.session_state.messages = []
                    st.session_state.dashboard_charts = []
                    st.session_state.data_warnings = validate_dataframe(df)
                    _audit("FILE_UPLOAD", f"name={uploaded.name} rows={len(df)} cols={len(df.columns)} mb={_mb:.1f}")
                    st.success(f"✅ {uploaded.name}")
                except Exception as e:
                    _audit("FILE_UPLOAD_ERROR", str(e))
                    st.error(str(e))

        # ── Demo button ────────────────────────────────────────────────────
        if st.button(T["demo_btn"], use_container_width=True):
            df = make_sample_df()
            tools.set_dataframe(df, name="sample_sales.csv")
            st.session_state.data_loaded = True
            st.session_state.chat = None   # rebuilt lazily in main()
            st.session_state._uploaded_file_id = None
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

            with st.expander(T["columns_expander"]):
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

        # ── AI Chart Style ─────────────────────────────────────────────────
        _STYLES  = ["whitegrid", "darkgrid", "white", "ticks"]
        _PALETTES = ["husl", "deep", "muted", "Set2", "tab10", "viridis", "rocket"]
        _SIZES   = {"10×5": (10, 5), "12×6": (12, 6), "8×4": (8, 4), "14×7": (14, 7)}
        with st.expander(T["chart_settings_hdr"], expanded=False):
            style_val = st.selectbox(
                T["chart_seaborn_style"], _STYLES,
                index=_STYLES.index(st.session_state.chart_seaborn_style),
                key="chart_style_sel",
            )
            pal_val = st.selectbox(
                T["chart_ai_palette"], _PALETTES,
                index=_PALETTES.index(st.session_state.chart_ai_palette),
                key="chart_pal_sel",
            )
            size_lbl = st.selectbox(
                T["chart_figsize"], list(_SIZES.keys()),
                index=list(_SIZES.keys()).index(st.session_state.chart_figsize_lbl),
                key="chart_size_sel",
            )
            st.session_state.chart_seaborn_style = style_val
            st.session_state.chart_ai_palette = pal_val
            st.session_state.chart_figsize_lbl = size_lbl
            tools.set_chart_style(
                seaborn_style=style_val,
                palette=pal_val,
                figsize=_SIZES[size_lbl],
            )

        st.markdown("---")

        # ── Email ──────────────────────────────────────────────────────────
        with st.expander(T["email_hdr"], expanded=False):
            st.caption(T["email_hint"])
            _efrom = st.text_input(
                T["email_from"], value=st.session_state.email_from,
                key="email_from_inp", placeholder="you@gmail.com",
            )
            _epwd = st.text_input(
                T["email_password"], value="",
                type="password", key="email_pwd_inp",
                placeholder="xxxx xxxx xxxx xxxx",
            )
            _eto = st.text_input(
                T["email_to"], value=st.session_state.email_to,
                key="email_to_inp", placeholder="recipient@example.com",
            )
            _esubj = st.text_input(
                T["email_subject"], value=st.session_state.email_subject,
                key="email_subj_inp",
            )
            _ewhat = st.radio(
                T["email_what"],
                options=["html", "csv"],
                format_func=lambda x: T["email_what_html"] if x == "html" else T["email_what_csv"],
                horizontal=True, key="email_what_radio",
            )
            if st.button(T["email_send"], use_container_width=True, key="email_send_btn"):
                # Persist non-sensitive fields
                st.session_state.email_from = _efrom
                st.session_state.email_to = _eto
                st.session_state.email_subject = _esubj
                st.session_state.email_what = _ewhat
                _df_now = tools.get_dataframe()
                try:
                    if _ewhat == "html":
                        if not st.session_state.messages:
                            st.warning(T["email_no_chat"])
                        else:
                            _body = export_chat_html(st.session_state.messages, T["page_title"]).decode("utf-8")
                            send_email_smtp(_efrom, _epwd, _eto, _esubj or T["page_title"], body_html=_body)
                            st.success(T["email_sent_ok"])
                    else:
                        if _df_now is None:
                            st.warning(T["email_no_data"])
                        else:
                            _csv_bytes = _df_now.to_csv(index=False).encode("utf-8-sig")
                            send_email_smtp(
                                _efrom, _epwd, _eto, _esubj or T["page_title"],
                                attachment_bytes=_csv_bytes, attachment_name="data.csv",
                            )
                            st.success(T["email_sent_ok"])
                except Exception as _ex:
                    st.error(T["email_sent_err"].format(err=str(_ex)))

        st.markdown("---")

        # ── What's New (Technical Writer) ──────────────────────────────────
        _cl_path = Path(__file__).parent / "CHANGELOG.md"
        if _cl_path.exists():
            with st.expander(T["whats_new_hdr"], expanded=False):
                _cl_lines = _cl_path.read_text(encoding="utf-8").splitlines()
                # Extract the first versioned section (between first ## and second ##)
                _section, _in = [], False
                for _line in _cl_lines:
                    if _line.startswith("## [") and not _line.startswith("## [Unreleased]"):
                        if _in:
                            break
                        _in = True
                    if _in:
                        _section.append(_line)
                st.markdown("\n".join(_section[:30]))

        st.markdown("---")

        # ── Clear ──────────────────────────────────────────────────────────
        if st.button(T["clear_btn"], use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat = None   # rebuilt lazily
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
    # Skip-to-main link for keyboard / screen-reader users
    st.markdown(
        f'<a class="skip-link" href="#main-content">{T["a11y_skip"]}</a>',
        unsafe_allow_html=True,
    )

    # ── Privacy Notice (Ethics — shown once per session) ──────────────────
    if not st.session_state._privacy_accepted:
        with st.container():
            st.info(T["privacy_notice"])
            if st.button(T["privacy_ok"], key="privacy_ok_btn"):
                st.session_state._privacy_accepted = True
                _audit("PRIVACY_ACCEPTED")
                st.rerun()

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
        if st.session_state.get("provider", "anthropic") == "ollama":
            _badge = st.session_state.get("ollama_model") or "🦙 ollama"
        else:
            _badge = "claude-opus-4-6"
        st.caption(_badge)

    # ── Sidebar ───────────────────────────────────────────────────────────
    render_sidebar(T)

    # ── Ensure chat object exists ─────────────────────────────────────────
    if st.session_state.chat is None and st.session_state.data_loaded:
        try:
            st.session_state.chat = build_chat()
        except ValueError:
            pass  # Ollama not configured yet — chat stays None

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
            if st.session_state.data_loaded:
                st.info(T["onboarding_hint"])
        else:
            # ── Export row ────────────────────────────────────────────────
            _has_charts = any(msg.get("charts") for msg in st.session_state.messages)
            _ec1, _ec2, _ec3, _ = st.columns([1, 1, 1, 3])
            with _ec1:
                st.download_button(
                    T["export_chat"],
                    data=export_chat_html(st.session_state.messages, T["page_title"]),
                    file_name="chat_export.html",
                    mime="text/html",
                    help=T["export_chat_help"],
                    use_container_width=True,
                    key="export_chat_btn",
                )
            with _ec2:
                st.download_button(
                    T["export_pdf"],
                    data=export_chat_pdf(st.session_state.messages, T["page_title"]),
                    file_name="chat_export.pdf",
                    mime="application/pdf",
                    help=T["export_pdf_help"],
                    use_container_width=True,
                    key="export_pdf_btn",
                )
            with _ec3:
                st.download_button(
                    T["export_ai_charts"],
                    data=export_ai_charts_zip(st.session_state.messages),
                    file_name="ai_charts.zip",
                    mime="application/zip",
                    help=T["export_ai_charts_help"],
                    use_container_width=True,
                    disabled=not _has_charts,
                    key="export_charts_btn",
                )
            render_history()

        # Resolve quick-question button presses
        pre_fill = st.session_state.pop("pending_input", None)
        if st.session_state.get("provider", "anthropic") == "ollama":
            no_key = not (st.session_state.ollama_available
                          and bool(st.session_state.ollama_model))
        else:
            no_key = not bool(get_active_key())
        no_data = tools.get_dataframe() is None
        user_input = st.chat_input(
            T["input_placeholder"],
            disabled=(no_key or no_data),
        )

        # Show contextual hint when chat input is disabled
        if no_data:
            st.caption(T["no_data_warn"])
        elif no_key:
            st.caption(T["no_key_hint"])

        # Guard: quick-question buttons must also respect disabled state
        final_input = (pre_fill if not (no_key or no_data) else None) or user_input

        _RATE_LIMIT = 30
        _MAX_INPUT  = 2000

        if final_input and st.session_state.chat:
            # ── Rate limit check ──────────────────────────────────────────
            if st.session_state._request_count >= _RATE_LIMIT:
                st.error(T["rate_limit_block"].format(limit=_RATE_LIMIT))
                final_input = None

        if final_input and st.session_state.chat:
            # ── Input sanitisation ────────────────────────────────────────
            if len(final_input) > _MAX_INPUT:
                st.caption(T["input_too_long"].format(max=_MAX_INPUT))
                final_input = final_input[:_MAX_INPUT]

            st.session_state._request_count += 1
            _req_n = st.session_state._request_count
            if _req_n >= _RATE_LIMIT * 0.8:    # warn at 80 %
                st.caption(T["rate_limit_warn"].format(n=_req_n, limit=_RATE_LIMIT))

            with st.chat_message("user"):
                st.markdown(final_input)
            st.session_state.messages.append({
                "role": "user", "content": final_input, "charts": [],
            })

            _t0 = datetime.datetime.now()
            with st.chat_message("assistant"):
                try:
                    full_text = st.write_stream(
                        text_stream(st.session_state.chat, final_input)
                    )
                except Exception as err:
                    full_text = ""
                    st.error(_friendly_error(err))
                    _audit("CHAT_ERROR", str(err))
                _elapsed = (datetime.datetime.now() - _t0).total_seconds()
                chart_paths = tools.get_pending_charts()
                for chart_path in chart_paths:
                    if chart_path.exists():
                        st.image(str(chart_path), use_container_width=True)

            _audit(
                "CHAT_RESPONSE",
                f"q_len={len(final_input)} resp_len={len(full_text or '')} "
                f"charts={len(chart_paths)} elapsed={_elapsed:.1f}s",
            )
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
                n_charts = len(st.session_state.dashboard_charts)
                st.success(f'{T["chart_added"]} {T["chart_added_total"].format(n=n_charts)}')

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
            hdr1, hdr2, hdr3, hdr4 = st.columns([3, 1, 1, 1])
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
            with hdr4:
                _dash_html = export_dashboard_html(
                    st.session_state.dashboard_charts, df_now, T["page_title"]
                )
                st.download_button(
                    T["export_dashboard"],
                    data=_dash_html.encode("utf-8"),
                    file_name="dashboard.html",
                    mime="text/html",
                    help=T["export_dashboard_help"],
                    use_container_width=True,
                    key="export_dash_btn",
                )

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

            cap_col, dl_col = st.columns([4, 1])
            with cap_col:
                st.caption(T["data_showing"].format(n=len(display_df), total=len(df_now)))
            with dl_col:
                csv_bytes = display_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label=T["export_csv"],
                    data=csv_bytes,
                    file_name=f"{tools.get_data_name() or 'data'}.csv",
                    mime="text/csv",
                    help=T["export_csv_help"],
                    use_container_width=True,
                )

            # ── Detailed stats ────────────────────────────────────────
            if st.checkbox(T["data_stats_chk"], key="data_stats_cb"):
                num_only = df_now.select_dtypes(include="number")
                if not num_only.empty:
                    st.dataframe(num_only.describe().round(2), use_container_width=True)
                else:
                    st.info(T["no_numeric_cols"])


if __name__ == "__main__":
    main()
