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
from chatlas import ChatAnthropic, ChatOpenAI, ChatGoogle, ChatGroq

# ─── Structured Logging (SRE) ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
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
        "theme_btn_light":  "☀️ מצב בהיר",
        "theme_btn_dark":   "🌙 מצב כהה",
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
        "tab_ai_dashboard":   "🤖 דשבורד AI",
        "tab_data":           "🗂 נתונים",
        "chat_welcome":           "שלום! הנתונים נטענו בהצלחה. שאל אותי כל שאלה על הנתונים שלך — אני אנתח, אצור גרפים ואתן תובנות.",
        # AI Dashboard
        "ai_dash_input_ph":       "תאר את הדשבורד שאתה רוצה...",
        "ai_dash_welcome":        "שלום! אני יכול לבנות דשבורד שלם על סמך תיאור שלך.\n\nלדוגמה:\n- \"בנה דשבורד מכירות עם גרפים לפי חודש, מוצר ואזור\"\n- \"הוסף גרף עוגה לקטגוריות\"\n- \"שנה את הגרף הראשון לגרף קו\"",
        "ai_dash_clear":          "🗑 נקה דשבורד",
        "ai_dash_clear_chat":     "🗑 נקה שיחה",
        "ai_dash_export":         "💾 ייצא",
        "ai_dash_charts_count":   "גרפים בדשבורד",
        "ai_dash_layout":         "פריסה",
        "ai_dash_copy_to_manual": "📋 העתק לדשבורד ידני",
        "ai_dash_copied":         "✅ הגרפים הועתקו!",
        "ai_dash_no_data":        "⬆️ העלה קובץ נתונים קודם",
        "ai_dash_empty_title":    "הדשבורד ריק",
        "ai_dash_empty_hint":     "תאר את הדשבורד שתרצה בשדה הקלט למטה",
        "ai_dash_remove_chart":   "× הסר",
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
        "provider_openai":    "🧠 OpenAI",
        "provider_google":    "✨ Google Gemini",
        "provider_groq":      "⚡ Groq",
        "provider_local":     "🦙 Ollama (מקומי)",
        "anthropic_model_lbl": "מודל Claude",
        "openai_model_lbl":   "מודל OpenAI",
        "google_model_lbl":   "מודל Gemini",
        "groq_model_lbl":     "מודל Groq",
        "ollama_model_lbl":   "מודל Ollama",
        "ollama_status_ok":   "✅ Ollama פעיל",
        "ollama_status_err":  "❌ Ollama לא נמצא",
        "ollama_hint":        "הורד והפעל Ollama: ollama.com",
        "ollama_no_models":   "לא נמצאו מודלים. הפעל: ollama pull llama3.2",
        "ollama_refresh":     "🔄 רענן",
        # הודעות נוספות
        "no_key_hint":        "⚠️ הגדר מפתח API או חבר Ollama",
        "columns_expander":   "עמודות",
        "data_showing":       "מציג {n:,} מתוך {total:,} שורות",
        "no_numeric_cols":    "אין עמודות מספריות לסטטיסטיקות",
        "stats_tab_numeric":  "מספרי",
        "stats_tab_categ":    "קטגוריאלי",
        "no_categ_cols":      "אין עמודות קטגוריאליות",
        "cat_order_hint":     "סמן Exclude להסרה, ערוך Order לשינוי סדר",
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
        "export_chat":           "📄 HTML",
        "export_chat_help":      "הורד את השיחה עם הגרפים כקובץ HTML",
        "export_dashboard":      "💾 דשבורד HTML",
        "export_dashboard_help": "הורד את כל הגרפים בדשבורד כ-HTML אינטראקטיבי",
        "export_ai_charts":      "🖼 גרפים ZIP",
        "export_ai_charts_help": "הורד את כל גרפי ה-AI שנוצרו בשיחה",
        "export_pdf":            "📑 PDF",
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
        # קוד + dashboard מה-chat
        "show_code":             "🔎 קוד",
        "add_to_dash_from_chat": "📌 הוסף לדשבורד",
        "chart_added_from_chat": "✅ גרף נוסף לדשבורד!",
        # דוח
        "export_report":         "📊 דוח HTML",
        "export_report_help":    "דוח מסודר של השיחה עם כל הממצאים והגרפים",
        # כפתורי export בתחתית הצ'אט
        "exports_lbl":           "⬇️ ייצוא שיחה",
        # inline upload (step 0)
        "upload_inline_title":   "👋 שלום! בואו נתחיל",
        "upload_inline_sub":     "העלה קובץ נתונים — CSV, Excel, JSON, Parquet",
        "upload_inline_or":      "— או —",
        "upload_inline_demo":    "🎲 נסה עם דאטה לדוגמה",
        # dataset summary (הודעה ראשונה)
        "ds_rows":               "שורות",
        "ds_cols":               "עמודות",
        "ds_numeric":            "מספריות",
        "ds_categorical":        "קטגוריות",
        "ds_dates":              "תאריכים",
        "ds_missing_ok":         "✅ אין ערכים חסרים",
        "ds_missing_warn":       "⚠️ {n:,} ערכים חסרים",
        "ds_prompt":             "💡 מה תרצה לנתח?",
        "ds_loaded":             "נטען!",
        "ds_data":               "נתונים",
        "ds_col_label":          "עמודות:",
        # chips
        "quick_chips_lbl":       "💡 שאלות להמשך:",
        # tabs חדשים בדשבורד
        "chart_builder_hdr":     "📊 בנה גרף ידני",
        "data_view_hdr":         "🗂 צפייה בנתונים",
        # p2 — thinking indicator + new-chat + dashboard empty state
        "thinking_indicator":    "מנתח...",
        "new_chat_btn":          "✨ שיחה חדשה",
        "new_chat_help":         "נקה את השיחה והתחל מחדש",
        "dash_empty_title":      "הדשבורד ריק כרגע",
        "dash_empty_step1":      "בחר עמודות ב<strong>בונה הגרפים</strong> למעלה",
        "dash_empty_step2":      "לחץ <strong>הוסף לדשבורד</strong> אחרי יצירת הגרף",
        "dash_empty_step3":      "או הוסף גרפים ישירות מהצ'אט עם הכפתור 📌",
        "dash_empty_tip":        "💡 טיפ: ניתן לבנות מספר גרפים ולסדר אותם בפריסה שונה",
        # guided tour
        "tour_restart":          "🎯 הדרכה",
        "tour_skip":             "✕ סגור",
        "tour_next":             "הבא ›",
        "tour_done_btn":         "✓ סיום",
        "tour_step_of":          "שלב {step} מתוך {total}",
        "tour_step1_title":      "📂 שלב 1 — העלאת נתונים",
        "tour_step1_body":       "לחץ על **'טען דאטה לדוגמה'** כדי להתחיל, או העלה קובץ CSV / Excel משלך.",
        "tour_step1_hint":       "⬇️ הכפתור נמצא בתיבת ההעלאה הגדולה",
        "tour_step2_title":      "💬 שלב 2 — שאל את ה-AI",
        "tour_step2_body":       "כתוב שאלה על הנתונים בתיבת הצ'אט, למשל: **'תראה לי סקירה'**.",
        "tour_step2_hint":       "⬇️ תיבת הצ'אט נמצאת בתחתית המסך",
        "tour_step3_title":      "📊 שלב 3 — בנה גרף בדשבורד",
        "tour_step3_body":       "לחץ על לשונית **'דשבורד'**, בחר גרף ועמודות, ולחץ **'הוסף לדשבורד'**.",
        "tour_step3_hint":       "⬆️ לחץ על לשונית 'דשבורד' למעלה",
        "tour_done_title":       "🎉 כל הכבוד — סיימת את ההדרכה!",
        "tour_done_body":        "עכשיו אתה מוכן! תוכל לייצא שיחות, לשלוח במייל, ולהוסיף עוד גרפים.",
        # code viewer panel
        "code_panel_lbl":        "🔎 קוד מאחורי התשובה",
        "code_tab_analysis":     "📊 שאילתת נתונים",
        "code_tab_chart":        "🎨 קוד יצירת הגרף",
        "code_panel_hint":       "לחץ על סמל ⎘ בפינת הקוד להעתקה",
        "code_toggle_help":      "הצג/הסתר קוד",
        "copy_response_help":    "העתק תשובה",
        "copy_response_ok":      "הועתק!",
        "chart_code_label":      "📋 קוד Python של הגרף",
        "chart_builder_err":     "שגיאה ביצירת גרף",
        "corr_matrix_title":     "מטריצת קורלציה",
        "unknown_chart_type":    "סוג גרף לא מוכר",
        # sidebar redesign — conversation history
        "new_conv_btn":          "✏️ שיחה חדשה",
        "conv_history_hdr":      "📜 שיחות",
        "settings_hdr":          "⚙️ הגדרות",
        "no_saved_chats":        "אין שיחות שמורות עדיין",
        "chat_loaded_note":      "📂 שיחה טעונה — ה-AI מתחיל הקשר חדש",
        "chat_date_today":       "היום",
        "chat_date_earlier":     "מוקדם יותר",
        "del_chat_btn":          "🗑",
        "del_chat_help":         "מחק שיחה זו",
        "dataset_badge_tpl":     "📊 {name} · {rows:,} שורות · {cols} עמודות",
        # chat-based onboarding
        "onboard_welcome": (
            "👋 שלום! אני מנתח הנתונים שלך.\n\n"
            "אני יכול לעזור לך:\n"
            "- 📊 לנתח ולסכם נתונים\n"
            "- 📈 ליצור גרפים ויזואליים\n"
            "- 🔍 לגלות תובנות ומגמות\n\n"
            "בוא נתחיל! ראשית, בחר כיצד להתחבר ל-AI:"
        ),
        "onboard_api_key_prompt": (
            "מעולה! הכנס את מפתח ה-API של Anthropic למטה.\n\n"
            "🔒 המפתח נשמר בזיכרון בלבד ולא נכתב לדיסק."
        ),
        "onboard_openai_prompt": (
            "מעולה! הכנס את מפתח ה-API של OpenAI למטה.\n\n"
            "🔒 המפתח נשמר בזיכרון בלבד ולא נכתב לדיסק."
        ),
        "onboard_google_prompt": (
            "מעולה! הכנס את מפתח ה-API של Google Gemini למטה.\n\n"
            "🔒 המפתח נשמר בזיכרון בלבד ולא נכתב לדיסק."
        ),
        "onboard_groq_prompt": (
            "מעולה! הכנס את מפתח ה-API של Groq למטה.\n\n"
            "🔒 המפתח נשמר בזיכרון בלבד ולא נכתב לדיסק."
        ),
        "onboard_ollama_prompt": "בוא נגדיר את חיבור Ollama המקומי.",
        "onboard_data_prompt": (
            "🎉 ה-AI מוכן! כעת העלה קובץ נתונים כדי להתחיל לנתח.\n\n"
            "אני תומך בקבצי CSV, Excel, JSON, TSV ו-Parquet."
        ),
        "api_hint_chat":       "🔗 קבל מפתח ב-console.anthropic.com",
        "api_invalid_key":     "המפתח חייב להתחיל ב-sk-ant-",
        "no_key_hint_chat":    "⚠️ הגדר מפתח API או חבר Ollama למעלה",
    },
    "en": {
        "page_title":       "🤖 Data Analyst Chatbot",
        "page_sub":         "Ask me anything about your data",
        "lang_btn":         "🇮🇱 עברית",
        "theme_btn_light":  "☀️ Light Mode",
        "theme_btn_dark":   "🌙 Dark Mode",
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
        "tab_ai_dashboard":   "🤖 AI Dashboard",
        "tab_data":           "🗂 Data",
        "chat_welcome":           "Hello! Your data is loaded. Ask me anything about your dataset — I'll analyze, create charts, and provide insights.",
        # AI Dashboard
        "ai_dash_input_ph":       "Describe the dashboard you want...",
        "ai_dash_welcome":        "Hello! I can build a complete dashboard from your description.\n\nExamples:\n- \"Build a sales dashboard with charts by month, product, and region\"\n- \"Add a pie chart for categories\"\n- \"Change the first chart to a line chart\"",
        "ai_dash_clear":          "🗑 Clear Dashboard",
        "ai_dash_clear_chat":     "🗑 Clear Chat",
        "ai_dash_export":         "💾 Export",
        "ai_dash_charts_count":   "charts in dashboard",
        "ai_dash_layout":         "Layout",
        "ai_dash_copy_to_manual": "📋 Copy to Manual Dashboard",
        "ai_dash_copied":         "✅ Charts copied!",
        "ai_dash_no_data":        "⬆️ Upload a data file first",
        "ai_dash_empty_title":    "Dashboard is empty",
        "ai_dash_empty_hint":     "Describe the dashboard you want in the input below",
        "ai_dash_remove_chart":   "× Remove",
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
        "provider_openai":    "🧠 OpenAI",
        "provider_google":    "✨ Google Gemini",
        "provider_groq":      "⚡ Groq",
        "provider_local":     "🦙 Ollama (local)",
        "anthropic_model_lbl": "Claude Model",
        "openai_model_lbl":   "OpenAI Model",
        "google_model_lbl":   "Gemini Model",
        "groq_model_lbl":     "Groq Model",
        "ollama_model_lbl":   "Ollama Model",
        "ollama_status_ok":   "✅ Ollama running",
        "ollama_status_err":  "❌ Ollama not found",
        "ollama_hint":        "Download & run Ollama: ollama.com",
        "ollama_no_models":   "No models found. Run: ollama pull llama3.2",
        "ollama_refresh":     "🔄 Refresh",
        # Extra strings
        "no_key_hint":        "⚠️ Configure API key or connect Ollama",
        "columns_expander":   "Columns",
        "data_showing":       "Showing {n:,} of {total:,} rows",
        "no_numeric_cols":    "No numeric columns for statistics",
        "stats_tab_numeric":  "Numeric",
        "stats_tab_categ":    "Categorical",
        "no_categ_cols":      "No categorical columns",
        "cat_order_hint":     "Check Exclude to remove, edit Order to reorder",
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
        "export_chat":           "📄 HTML",
        "export_chat_help":      "Download the conversation with charts as an HTML file",
        "export_dashboard":      "💾 Dashboard HTML",
        "export_dashboard_help": "Download all dashboard charts as an interactive HTML file",
        "export_ai_charts":      "🖼 Charts ZIP",
        "export_ai_charts_help": "Download all AI-generated charts from this conversation",
        "export_pdf":            "📑 PDF",
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
        # code + dashboard from chat
        "show_code":             "🔎 Code",
        "add_to_dash_from_chat": "📌 Add to Dashboard",
        "chart_added_from_chat": "✅ Chart added to Dashboard!",
        # report
        "export_report":         "📊 Report HTML",
        "export_report_help":    "Structured report of the conversation with all findings and charts",
        # export buttons label
        "exports_lbl":           "⬇️ Export conversation",
        # inline upload (step 0)
        "upload_inline_title":   "👋 Welcome! Let's get started",
        "upload_inline_sub":     "Upload a data file — CSV, Excel, JSON, Parquet",
        "upload_inline_or":      "— or —",
        "upload_inline_demo":    "🎲 Try with sample data",
        # dataset summary
        "ds_rows":               "Rows",
        "ds_cols":               "Columns",
        "ds_numeric":            "Numeric",
        "ds_categorical":        "Categorical",
        "ds_dates":              "Dates",
        "ds_missing_ok":         "✅ No missing values",
        "ds_missing_warn":       "⚠️ {n:,} missing values",
        "ds_prompt":             "💡 What would you like to analyse?",
        "ds_loaded":             "loaded!",
        "ds_data":               "Data",
        "ds_col_label":          "Columns:",
        # chips
        "quick_chips_lbl":       "💡 Suggested questions:",
        # dashboard tab sections
        "chart_builder_hdr":     "📊 Build a Chart",
        "data_view_hdr":         "🗂 Data View",
        # p2 — thinking indicator + new-chat + dashboard empty state
        "thinking_indicator":    "Analyzing...",
        "new_chat_btn":          "✨ New Chat",
        "new_chat_help":         "Clear conversation and start fresh",
        "dash_empty_title":      "Your dashboard is empty",
        "dash_empty_step1":      "Choose columns in the <strong>Chart Builder</strong> above",
        "dash_empty_step2":      "Click <strong>Add to Dashboard</strong> after building a chart",
        "dash_empty_step3":      "Or add charts directly from the AI chat using the 📌 button",
        "dash_empty_tip":        "💡 Tip: Build multiple charts and arrange them in different layouts",
        # guided tour
        "tour_restart":          "🎯 Tour",
        "tour_skip":             "✕ Close",
        "tour_next":             "Next ›",
        "tour_done_btn":         "✓ Done",
        "tour_step_of":          "Step {step} of {total}",
        "tour_step1_title":      "📂 Step 1 — Upload Data",
        "tour_step1_body":       "Click **'Load Sample Data'** to get started, or upload your own CSV / Excel file.",
        "tour_step1_hint":       "⬇️ The button is inside the large upload zone below",
        "tour_step2_title":      "💬 Step 2 — Ask the AI",
        "tour_step2_body":       "Type a question about your data in the chat box, e.g. **'Give me a data overview'**.",
        "tour_step2_hint":       "⬇️ The chat input is at the bottom of the screen",
        "tour_step3_title":      "📊 Step 3 — Build a Dashboard Chart",
        "tour_step3_body":       "Click the **'Dashboard'** tab, pick chart type and columns, then click **'Add to Dashboard'**.",
        "tour_step3_hint":       "⬆️ Click the 'Dashboard' tab above",
        "tour_done_title":       "🎉 Great job — you finished the tour!",
        "tour_done_body":        "You're all set! You can export chats, send via email, and build more charts.",
        # code viewer panel
        "code_panel_lbl":        "🔎 Code behind this answer",
        "code_tab_analysis":     "📊 Data Query",
        "code_tab_chart":        "🎨 Chart Code",
        "code_panel_hint":       "Click the ⎘ icon in the code corner to copy",
        "code_toggle_help":      "Show/hide code",
        "copy_response_help":    "Copy response",
        "copy_response_ok":      "Copied!",
        "chart_code_label":      "📋 Python chart code",
        "chart_builder_err":     "Error building chart",
        "corr_matrix_title":     "Correlation Matrix",
        "unknown_chart_type":    "Unknown chart type",
        # sidebar redesign — conversation history
        "new_conv_btn":          "✏️ New Conversation",
        "conv_history_hdr":      "📜 Conversations",
        "settings_hdr":          "⚙️ Settings",
        "no_saved_chats":        "No saved conversations yet",
        "chat_loaded_note":      "📂 Conversation loaded — AI is starting a fresh context",
        "chat_date_today":       "Today",
        "chat_date_earlier":     "Earlier",
        "del_chat_btn":          "🗑",
        "del_chat_help":         "Delete this conversation",
        "dataset_badge_tpl":     "📊 {name} · {rows:,} rows · {cols} columns",
        # chat-based onboarding
        "onboard_welcome": (
            "👋 Hello! I'm your AI Data Analyst.\n\n"
            "I can help you:\n"
            "- 📊 Analyze and summarize data\n"
            "- 📈 Create visualizations\n"
            "- 🔍 Discover insights and trends\n\n"
            "Let's get started! First, choose how to connect to AI:"
        ),
        "onboard_api_key_prompt": (
            "Great! Enter your Anthropic API key below.\n\n"
            "🔒 Your key is stored in memory only and is never saved to disk."
        ),
        "onboard_openai_prompt": (
            "Great! Enter your OpenAI API key below.\n\n"
            "🔒 Your key is stored in memory only and is never saved to disk."
        ),
        "onboard_google_prompt": (
            "Great! Enter your Google Gemini API key below.\n\n"
            "🔒 Your key is stored in memory only and is never saved to disk."
        ),
        "onboard_groq_prompt": (
            "Great! Enter your Groq API key below.\n\n"
            "🔒 Your key is stored in memory only and is never saved to disk."
        ),
        "onboard_ollama_prompt": "Let's set up your local Ollama connection.",
        "onboard_data_prompt": (
            "🎉 AI is ready! Now upload a data file to start analyzing.\n\n"
            "I support CSV, Excel, JSON, TSV, and Parquet files."
        ),
        "api_hint_chat":       "🔗 Get your key at console.anthropic.com",
        "api_invalid_key":     "Key must start with sk-ant-",
        "no_key_hint_chat":    "⚠️ Set up API key or connect Ollama above",
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

# ─── AI Dashboard System Prompt ──────────────────────────────────────────────

AI_DASHBOARD_SYSTEM_PROMPT = """
You are an expert Dashboard Designer assistant.
Respond in the same language the user writes in (Hebrew or English).

## Your Role
You help users build complete interactive dashboards by generating chart configurations.
You do NOT create individual analyses or text reports — you ONLY build dashboards.

## Process
1. FIRST call get_data_overview() to understand the dataset columns and types.
2. Based on the user's description, decide which charts best tell the data story.
3. Call set_dashboard_charts() with a list of chart config dicts to create the full dashboard.
4. To modify an existing chart: call update_dashboard_chart(index, updates).
5. To add a chart: call add_dashboard_chart(chart).
6. To remove a chart: call remove_dashboard_chart(index).

## Chart Config Format
Each chart config is a dict with these keys:
- "type": one of "Bar", "Line", "Area", "Pie", "Histogram", "Scatter", "Box Plot", "Heatmap"
          (Short forms like "bar", "line", "pie" are also accepted)
- "x": column name for X-axis (MUST be an actual column in the dataset)
- "y": column name for Y-axis (numeric column; use "—" if not needed, e.g. for Pie)
- "title": descriptive chart title in the user's language
- "color": (optional) column name for color grouping, or null
- "palette": (optional) one of "Blue", "Purple", "Green", "Orange", "Pink", "Teal"
- "sample_size": (optional) number of rows to plot, default 500
- "corr_method": (optional, Heatmap only) "pearson" | "spearman" | "kendall"

## Guidelines
- Aim for 3-6 charts per dashboard that tell a coherent data story.
- Vary chart types for visual diversity — don't create 6 bar charts.
- Use meaningful, descriptive titles in the user's language.
- Assign different color palettes to different charts for visual variety.
- When the user asks to modify: change ONLY the requested chart(s), preserve everything else.
- When the user says "first chart" or "chart 1", that means index 0.

## Response Format
After calling the tool(s), respond with a SHORT summary:
1. What charts were created or modified
2. What story the dashboard tells about the data
3. 1-2 suggested improvements the user could request

## Guardrails (STRICT)
- ONLY use column names that actually exist in the dataset (verify with get_data_overview).
- NEVER invent column names or data.
- NEVER suggest modifying the user's source data.
- NEVER execute code — use only the dashboard tools.
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


# ─── Provider Registry ────────────────────────────────────────────────────────
PROVIDER_REGISTRY = {
    "anthropic": {
        "label_key":     "provider_cloud",
        "key_prefix":    "sk-ant-",
        "key_session":   "api_key",
        "env_var":       "ANTHROPIC_API_KEY",
        "default_model": "claude-opus-4-6",
        "models":        ["claude-opus-4-6", "claude-sonnet-4-20250514"],
        "model_session": "anthropic_model",
        "placeholder":   "sk-ant-...",
        "hint_url":      "console.anthropic.com",
        "max_tokens":    8192,
    },
    "openai": {
        "label_key":     "provider_openai",
        "key_prefix":    "sk-",
        "key_session":   "openai_api_key",
        "env_var":       "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "models":        ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o3-mini"],
        "model_session": "openai_model",
        "placeholder":   "sk-...",
        "hint_url":      "platform.openai.com/api-keys",
        "max_tokens":    4096,
    },
    "google": {
        "label_key":     "provider_google",
        "key_prefix":    "AIza",
        "key_session":   "google_api_key",
        "env_var":       "GOOGLE_API_KEY",
        "default_model": "gemini-2.0-flash",
        "models":        ["gemini-2.0-flash", "gemini-2.5-pro", "gemini-1.5-pro"],
        "model_session": "google_model",
        "placeholder":   "AIza...",
        "hint_url":      "aistudio.google.com/apikey",
        "max_tokens":    None,
    },
    "groq": {
        "label_key":     "provider_groq",
        "key_prefix":    "gsk_",
        "key_session":   "groq_api_key",
        "env_var":       "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        "models":        ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
        "model_session": "groq_model",
        "placeholder":   "gsk_...",
        "hint_url":      "console.groq.com/keys",
        "max_tokens":    4096,
    },
    "ollama": {
        "label_key":     "provider_local",
        "key_prefix":    None,
        "key_session":   None,
        "env_var":       None,
        "default_model": "",
        "models":        [],
        "model_session": "ollama_model",
        "placeholder":   None,
        "hint_url":      "ollama.com",
        "max_tokens":    None,
    },
}


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
    # ── Sequential (original) ──
    "סגול / Purple": {"single": "#7c6af7", "seq": px.colors.sequential.Purp,    "scale": "Purp"},
    "כחול / Blue":   {"single": "#1a73e8", "seq": px.colors.sequential.Blues,   "scale": "Blues"},
    "ירוק / Green":  {"single": "#2ecc71", "seq": px.colors.sequential.Greens,  "scale": "Greens"},
    "כתום / Orange": {"single": "#f39c12", "seq": px.colors.sequential.Oranges, "scale": "Oranges"},
    "ורוד / Pink":   {"single": "#f76a8a", "seq": px.colors.sequential.RdPu,    "scale": "RdPu"},
    "ציאן / Teal":   {"single": "#00bcd4", "seq": px.colors.sequential.Teal,    "scale": "Teal"},
    # ── Sequential (new) ──
    "שקיעה / Sunset":  {"single": "#ff6b6b", "seq": px.colors.sequential.Sunsetdark, "scale": "Sunsetdark"},
    "אפור / Gray":     {"single": "#888888", "seq": px.colors.sequential.gray,        "scale": "gray"},
    "מג'נטה / Magenta": {"single": "#e040fb", "seq": px.colors.sequential.Magenta,    "scale": "Magenta"},
    # ── Diverging (ideal for heatmaps with +/- values) ──
    "כחול-אדום / RdBu":  {"single": "#3b7dd8", "seq": px.colors.diverging.RdBu,     "scale": "RdBu"},
    "חם-קר / Spectral":   {"single": "#d73027", "seq": px.colors.diverging.Spectral,  "scale": "Spectral"},
    "ירוק-סגול / PRGn":  {"single": "#1b7837", "seq": px.colors.diverging.PRGn,      "scale": "PRGn"},
    # ── Qualitative (best for categorical data with many groups) ──
    "צבעוני / Vivid":  {"single": "#e45756", "seq": px.colors.qualitative.Vivid,   "scale": "Vivid"},
    "בולד / Bold":     {"single": "#7c4dff", "seq": px.colors.qualitative.Bold,    "scale": "Bold"},
    "פסטל / Pastel":   {"single": "#aec7e8", "seq": px.colors.qualitative.Pastel,  "scale": "Pastel"},
    "D3 / D3":         {"single": "#1f77b4", "seq": px.colors.qualitative.D3,      "scale": "D3"},
}

CHART_TYPES = [
    "עמודות / Bar", "קו / Line", "שטח / Area", "עוגה / Pie",
    "היסטוגרמה / Histogram", "פיזור / Scatter", "Box Plot", "Heatmap",
]

# ── Normalisation maps: accept English-only, Hebrew-only or bilingual keys ───
_CHART_KEY_MAP: dict[str, str] = {}
for _ct in CHART_TYPES:
    _CHART_KEY_MAP[_ct] = _ct
    _CHART_KEY_MAP[_ct.lower()] = _ct
    if " / " in _ct:
        _he, _en = _ct.split(" / ", 1)
        _CHART_KEY_MAP[_he] = _ct
        _CHART_KEY_MAP[_en] = _ct
        _CHART_KEY_MAP[_he.lower()] = _ct
        _CHART_KEY_MAP[_en.lower()] = _ct

_PALETTE_KEY_MAP: dict[str, str] = {}
for _pk in COLOR_PALETTES:
    _PALETTE_KEY_MAP[_pk] = _pk
    if " / " in _pk:
        _he, _en = _pk.split(" / ", 1)
        _PALETTE_KEY_MAP[_he] = _pk
        _PALETTE_KEY_MAP[_en] = _pk


def _norm_chart_type(ct: str) -> str:
    """Map any chart type string to the internal bilingual key."""
    return _CHART_KEY_MAP.get(ct, _CHART_KEY_MAP.get(ct.strip().lower(), ct))


def _norm_palette(p: str) -> str:
    """Map any palette name to the internal bilingual key."""
    return _PALETTE_KEY_MAP.get(p, _PALETTE_KEY_MAP.get(p.strip(), p))


def _chart_types_for_lang() -> list[str]:
    """Return chart type labels for current UI language."""
    if st.session_state.get("lang") == "en":
        return [ct.split(" / ")[-1] if " / " in ct else ct for ct in CHART_TYPES]
    return list(CHART_TYPES)


def _palettes_for_lang() -> dict[str, dict]:
    """Return palette dict with display keys for current UI language."""
    if st.session_state.get("lang") == "en":
        return {(k.split(" / ")[-1] if " / " in k else k): v
                for k, v in COLOR_PALETTES.items()}
    return dict(COLOR_PALETTES)


# ─── Theme Color Tokens ─────────────────────────────────────────────────────
THEME_COLORS = {
    "dark": {
        # Text
        "text-primary": "#ececec", "text-secondary": "#b4b4b4",
        "text-tertiary": "#999", "text-muted": "#888", "text-disabled": "#555",
        # Backgrounds
        "bg-app": "#212121", "bg-sidebar": "#171717",
        "bg-surface": "#2f2f2f", "bg-surface-raised": "#303030",
        "bg-plot": "#2a2a2a", "bg-code": "#1e1e1e",
        "bg-hover": "rgba(255,255,255,0.08)",
        "bg-legend": "rgba(255,255,255,0.07)",
        # Borders
        "border-default": "#383838", "border-strong": "#4e4e4e",
        "border-subtle": "#2a2a2a",
        # Accent
        "accent": "#10a37f", "accent-hover": "#0d8c6b",
        "accent-bg": "#2a3a2e",
        # Status badges
        "status-ok-bg": "#1a3a2a", "status-ok-text": "#56d364",
        "status-ok-border": "#2ea043",
        "status-err-bg": "#3a1a1a", "status-err-text": "#f85149",
        "status-err-border": "#da3633",
        # Plotly charts
        "chart-plot": "#2a2a2a", "chart-grid": "#383838",
        "chart-zeroline": "#4e4e4e", "chart-axis": "#b4b4b4",
        "chart-spine": "#4e4e4e", "chart-hover-bg": "#2f2f2f",
        # Scrollbar
        "scrollbar-thumb": "#4e4e4e", "scrollbar-hover": "#666",
        # Chat input (Calcalist-style)
        "input-bg": "#343541", "input-border": "#565869",
        "input-shadow": "0 4px 16px rgba(0,0,0,0.35)",
        "input-focus-border": "#10a37f",
        "input-focus-shadow": "0 4px 20px rgba(16,163,127,0.25)",
        "input-placeholder": "#8e8ea0",
        # Gradients
        "welcome-gradient": "linear-gradient(135deg, #10a37f08, #10a37f04)",
        "input-gradient": "linear-gradient(to bottom, transparent, #212121 40%)",
    },
    "light": {
        # Text
        "text-primary": "#1a1a1a", "text-secondary": "#555",
        "text-tertiary": "#777", "text-muted": "#888", "text-disabled": "#bbb",
        # Backgrounds
        "bg-app": "#ffffff", "bg-sidebar": "#f5f5f5",
        "bg-surface": "#f0f0f0", "bg-surface-raised": "#e8e8e8",
        "bg-plot": "#fafafa", "bg-code": "#f5f5f5",
        "bg-hover": "rgba(0,0,0,0.05)",
        "bg-legend": "rgba(0,0,0,0.04)",
        # Borders
        "border-default": "#ddd", "border-strong": "#ccc",
        "border-subtle": "#e5e5e5",
        # Accent (same teal — good contrast in both modes)
        "accent": "#10a37f", "accent-hover": "#0d8c6b",
        "accent-bg": "#e6f5ef",
        # Status badges
        "status-ok-bg": "#e6f5ef", "status-ok-text": "#1a7f37",
        "status-ok-border": "#2ea043",
        "status-err-bg": "#fde8e8", "status-err-text": "#cf222e",
        "status-err-border": "#da3633",
        # Plotly charts
        "chart-plot": "#fafafa", "chart-grid": "#e0e0e0",
        "chart-zeroline": "#ccc", "chart-axis": "#555",
        "chart-spine": "#ccc", "chart-hover-bg": "#ffffff",
        # Scrollbar
        "scrollbar-thumb": "#ccc", "scrollbar-hover": "#aaa",
        # Chat input (Calcalist-style)
        "input-bg": "#ffffff", "input-border": "#d9d9e3",
        "input-shadow": "0 4px 16px rgba(0,0,0,0.10)",
        "input-focus-border": "#10a37f",
        "input-focus-shadow": "0 4px 20px rgba(16,163,127,0.18)",
        "input-placeholder": "#8e8ea0",
        # Gradients
        "welcome-gradient": "linear-gradient(135deg, #10a37f10, #10a37f06)",
        "input-gradient": "linear-gradient(to bottom, transparent, #ffffff 40%)",
    },
}


def apply_chart_style(fig: go.Figure, theme: str = "dark") -> go.Figure:
    """Apply a polished, modern visual style to any Plotly figure."""
    tc = THEME_COLORS[theme]
    _PAPER = "rgba(0,0,0,0)"   # transparent — inherits Streamlit's background
    _PLOT  = tc["chart-plot"]
    _GRID  = tc["chart-grid"]
    _ZERO  = tc["chart-zeroline"]
    _TEXT  = tc["text-primary"]
    _AXIS  = tc["chart-axis"]
    _LINE  = tc["chart-spine"]
    _HOVER = tc["chart-hover-bg"]
    _lang  = st.session_state.get("lang", "he")

    # Preserve any existing title text; if none, use "" to prevent Plotly.js "undefined" rendering
    _title_text = getattr(getattr(fig.layout, "title", None), "text", None) or ""

    fig.update_layout(
        height=480,
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PLOT,
        margin=dict(t=64, b=52, l=60, r=24, pad=4),
        font=dict(
            family="Inter, Segoe UI, system-ui, -apple-system, sans-serif",
            size=13,
            color=_TEXT,
        ),
        # ── Typography hierarchy ──
        title=dict(
            text=_title_text,
            font=dict(size=17, color=_TEXT),
            x=0.02,
            xanchor="left",
            yanchor="top",
            pad=dict(b=12),
        ),
        # ── Legend: horizontal, below chart, transparent ──
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=11, color=tc["text-secondary"]),
            itemsizing="constant",
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            itemwidth=30,
        ),
        # ── X axis ──
        xaxis=dict(
            showgrid=True,
            gridcolor=_GRID,
            gridwidth=1,
            griddash="dot",
            zeroline=True,
            zerolinecolor=_ZERO,
            zerolinewidth=1.5,
            linecolor=_LINE,
            linewidth=1,
            tickfont=dict(size=11, color=_AXIS),
            title_font=dict(size=12, color=_AXIS),
            title_standoff=12,
            automargin=True,
            ticklabeloverflow="allow",
            separatethousands=True,
        ),
        # ── Y axis ──
        yaxis=dict(
            showgrid=True,
            gridcolor=_GRID,
            gridwidth=1,
            griddash="dot",
            zeroline=True,
            zerolinecolor=_ZERO,
            zerolinewidth=1.5,
            linecolor=_LINE,
            linewidth=1,
            tickfont=dict(size=11, color=_AXIS),
            title_font=dict(size=12, color=_AXIS),
            title_standoff=12,
            automargin=True,
            separatethousands=True,
        ),
        # ── Hover ──
        hoverlabel=dict(
            bgcolor=_HOVER,
            bordercolor=_GRID,
            font_size=13,
            font_color=_TEXT,
            font_family="Inter, Segoe UI, sans-serif",
            align="right" if _lang == "he" else "left",
            namelength=-1,
        ),
        # ── Colorbar for heatmaps ──
        coloraxis_colorbar=dict(
            tickfont=dict(color=_AXIS, size=11),
            title_font=dict(color=_AXIS, size=12),
            thickness=14,
            len=0.8,
            outlinewidth=0,
        ),
        # ── Smooth transitions ──
        transition=dict(
            duration=500,
            easing="cubic-in-out",
            ordering="traces first",
        ),
    )

    # ── Auto-rotate x labels when many categories ──
    x_data = fig.data[0].x if fig.data and hasattr(fig.data[0], "x") and fig.data[0].x is not None else []
    if hasattr(x_data, "__len__") and len(x_data) > 8:
        fig.update_layout(xaxis_tickangle=-45)
    # ── Trace-level refinements ──
    # Bar: clean borders, slight transparency
    fig.update_traces(
        selector=dict(type="bar"),
        marker_line_width=0,
        opacity=0.92,
    )
    # Scatter: refined markers with subtle outline
    fig.update_traces(
        selector=dict(type="scatter", mode="markers"),
        marker=dict(size=8, opacity=0.8, line=dict(width=1, color="rgba(255,255,255,0.3)")),
    )
    # Line traces: thicker line
    fig.update_traces(
        selector=dict(type="scatter", mode="lines"),
        line=dict(width=2.5),
    )
    # Pie: modern donut with subtle slice separation
    fig.update_traces(
        selector=dict(type="pie"),
        hole=0.4,
        textfont=dict(size=12, color=_TEXT),
        textposition="auto",
        pull=[0.02] * 20,
    )

    # ── Hover templates per trace type ──
    fig.update_traces(
        selector=dict(type="bar"),
        hovertemplate="<b>%{x}</b><br>%{y:,.0f}<extra></extra>",
    )
    fig.update_traces(
        selector=dict(type="scatter", mode="markers"),
        hovertemplate="<b>%{x}</b><br>Y: %{y:,.2f}<extra></extra>",
    )
    fig.update_traces(
        selector=dict(type="scatter", mode="lines"),
        hovertemplate="<b>%{x}</b><br>%{y:,.2f}<extra></extra>",
    )
    fig.update_traces(
        selector=dict(type="pie"),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
    )
    fig.update_traces(
        selector=dict(type="histogram"),
        hovertemplate="%{x}<br>Count: %{y:,}<extra></extra>",
    )
    fig.update_traces(
        selector=dict(type="heatmap"),
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Value: %{z:.3f}<extra></extra>",
    )

    return fig


def build_chart(config: dict, df: pd.DataFrame, sample_size: int = 500) -> go.Figure:
    """Build a Plotly figure from a chart-config dict."""
    chart_type   = _norm_chart_type(config.get("type", CHART_TYPES[0]))
    x_col        = config.get("x")
    y_col        = config.get("y", "—")
    color        = config.get("color")
    title        = config.get("title", "")
    palette_name = _norm_palette(config.get("palette", list(COLOR_PALETTES.keys())[0]))
    corr_method  = config.get("corr_method", "pearson")
    palette      = COLOR_PALETTES.get(palette_name, list(COLOR_PALETTES.values())[0])

    num_cols = df.select_dtypes(include="number").columns.tolist()
    common   = {"title": title} if title else {}
    single   = palette["single"]
    seq      = palette["seq"]
    dfs      = df.head(sample_size)

    # Extract English type key (handles "עמודות / Bar" → "Bar" and "Box Plot" → "Box Plot")
    t = chart_type.split(" / ")[-1]

    try:
        if t == "Bar":
            if y_col and y_col != "—":
                fig = px.bar(dfs, x=x_col, y=y_col, color=color,
                             color_discrete_sequence=seq, **common)
            else:
                counts = df[x_col].value_counts().head(20).reset_index()
                counts.columns = [x_col, "count"]
                fig = px.bar(counts, x=x_col, y="count",
                             color_discrete_sequence=[single], **common)

        elif t == "Line":
            y = y_col if y_col and y_col != "—" else (num_cols[0] if num_cols else x_col)
            fig = px.line(dfs, x=x_col, y=y, color=color,
                          color_discrete_sequence=[single], **common)

        elif t == "Area":
            y = y_col if y_col and y_col != "—" else (num_cols[0] if num_cols else x_col)
            fig = px.area(dfs, x=x_col, y=y, color=color,
                          color_discrete_sequence=[single], **common)

        elif t == "Pie":
            counts = df[x_col].value_counts().head(10).reset_index()
            counts.columns = [x_col, "count"]
            fig = px.pie(counts, names=x_col, values="count",
                         color_discrete_sequence=seq, **common)

        elif t == "Histogram":
            col = y_col if y_col and y_col != "—" else (num_cols[0] if num_cols else x_col)
            fig = px.histogram(df, x=col, color_discrete_sequence=[single], **common)

        elif t == "Scatter":
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
                title=title or f'{TEXT[st.session_state.get("lang", "he")]["corr_matrix_title"]} ({corr_method})',
                text_auto=True,
            )

        else:
            fig = go.Figure()
            _T = TEXT[st.session_state.get("lang", "he")]
            fig.add_annotation(text=f'{_T["unknown_chart_type"]}: {chart_type}', showarrow=False)

        return apply_chart_style(fig, theme=st.session_state.get("theme", "dark"))

    except Exception as exc:
        err_fig = go.Figure()
        _T = TEXT[st.session_state.get("lang", "he")]
        err_fig.add_annotation(text=f'{_T["chart_builder_err"]}: {exc}', showarrow=False,
                               font={"color": "#c5221f"})
        return apply_chart_style(err_fig, theme=st.session_state.get("theme", "dark"))


def _build_chart_code(config: dict, sample_size: int = 500) -> str:
    """Generate equivalent Python/Plotly code for a chart config dict."""
    ct           = _norm_chart_type(config.get("type", CHART_TYPES[0]))
    x            = config.get("x", "")
    y            = config.get("y", "—")
    color        = config.get("color")
    title        = config.get("title", "")
    pal_name     = _norm_palette(config.get("palette", list(COLOR_PALETTES.keys())[0]))
    corr_method  = config.get("corr_method", "pearson")
    t            = ct.split(" / ")[-1]
    palette      = COLOR_PALETTES.get(pal_name, list(COLOR_PALETTES.values())[0])
    single       = repr(palette["single"])
    _PAL_SEQ = {
        # Sequential
        "סגול / Purple": "px.colors.sequential.Purp",
        "כחול / Blue":   "px.colors.sequential.Blues",
        "ירוק / Green":  "px.colors.sequential.Greens",
        "כתום / Orange": "px.colors.sequential.Oranges",
        "ורוד / Pink":   "px.colors.sequential.RdPu",
        "ציאן / Teal":   "px.colors.sequential.Teal",
        "שקיעה / Sunset":  "px.colors.sequential.Sunsetdark",
        "אפור / Gray":     "px.colors.sequential.gray",
        "מג'נטה / Magenta": "px.colors.sequential.Magenta",
        # Diverging
        "כחול-אדום / RdBu":  "px.colors.diverging.RdBu",
        "חם-קר / Spectral":   "px.colors.diverging.Spectral",
        "ירוק-סגול / PRGn":  "px.colors.diverging.PRGn",
        # Qualitative
        "צבעוני / Vivid":  "px.colors.qualitative.Vivid",
        "בולד / Bold":     "px.colors.qualitative.Bold",
        "פסטל / Pastel":   "px.colors.qualitative.Pastel",
        "D3 / D3":         "px.colors.qualitative.D3",
    }
    seq          = _PAL_SEQ.get(pal_name, "px.colors.sequential.Blues")
    scale        = repr(palette.get("scale", "Blues"))
    title_arg    = f', title="{title}"' if title else ""
    color_arg    = f', color="{color}"' if color else ""

    lines = [
        "import plotly.express as px",
        "import plotly.graph_objects as go",
        "",
        "# df  ← your pandas DataFrame",
    ]

    if t == "Bar":
        if y and y != "—":
            lines += [
                f"dfs = df.head({sample_size})",
                f'fig = px.bar(dfs, x="{x}", y="{y}"{color_arg},',
                f'             color_discrete_sequence={seq}{title_arg})',
            ]
        else:
            lines += [
                f'counts = df["{x}"].value_counts().head(20).reset_index()',
                f'counts.columns = ["{x}", "count"]',
                f'fig = px.bar(counts, x="{x}", y="count",',
                f'             color_discrete_sequence=[{single}]{title_arg})',
            ]
    elif t == "Line":
        y_use = y if y and y != "—" else x
        lines += [
            f"dfs = df.head({sample_size})",
            f'fig = px.line(dfs, x="{x}", y="{y_use}"{color_arg},',
            f'              color_discrete_sequence=[{single}]{title_arg})',
        ]
    elif t == "Area":
        y_use = y if y and y != "—" else x
        lines += [
            f"dfs = df.head({sample_size})",
            f'fig = px.area(dfs, x="{x}", y="{y_use}"{color_arg},',
            f'              color_discrete_sequence=[{single}]{title_arg})',
        ]
    elif t == "Pie":
        lines += [
            f'counts = df["{x}"].value_counts().head(10).reset_index()',
            f'counts.columns = ["{x}", "count"]',
            f'fig = px.pie(counts, names="{x}", values="count",',
            f'             color_discrete_sequence={seq}{title_arg})',
        ]
    elif t == "Histogram":
        col = y if y and y != "—" else x
        lines += [
            f'fig = px.histogram(df, x="{col}",',
            f'                   color_discrete_sequence=[{single}]{title_arg})',
        ]
    elif t == "Scatter":
        y_use = y if y and y != "—" else x
        lines += [
            f"dfs = df.head({sample_size})",
            f'fig = px.scatter(dfs, x="{x}", y="{y_use}"{color_arg},',
            f'                 color_discrete_sequence=[{single}]{title_arg})',
        ]
    elif t == "Box Plot":
        y_use = y if y and y != "—" else x
        x_arg = f'x="{x}", ' if x != y_use else ""
        lines += [
            f'fig = px.box(df, {x_arg}y="{y_use}"{color_arg},',
            f'             color_discrete_sequence=[{single}]{title_arg})',
        ]
    elif t == "Heatmap":
        lines += [
            f'corr = df.select_dtypes(include="number").corr(method="{corr_method}").round(3)',
            f'fig = px.imshow(corr, color_continuous_scale={scale},',
            f'                text_auto=True{title_arg})',
        ]
    else:
        lines += [f"fig = go.Figure()  # unknown chart type"]

    lines += ["", "fig.show()"]
    return "\n".join(lines)


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

    try:
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
        page_title_w = pdf.w - pdf.l_margin - pdf.r_margin
        pdf.set_font(fnt, size=18)
        pdf.multi_cell(page_title_w, 10, _fix(title), align="R" if _is_rtl(title) else "L")
        pdf.ln(4)

        # Usable page width (same value used for all multi_cell calls)
        page_w = pdf.w - pdf.l_margin - pdf.r_margin

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            charts = msg.get("charts", [])

            # Role label
            pdf.set_font(fnt, size=9)
            if role == "assistant":
                pdf.set_fill_color(230, 240, 255)
            else:
                pdf.set_fill_color(230, 250, 230)
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
                if not fixed:
                    continue
                try:
                    pdf.multi_cell(page_w, 5, fixed, align="R" if _is_rtl(clean) else "L")
                except Exception:
                    try:
                        pdf.multi_cell(page_w, 5, "[...]", align="L")
                    except Exception:
                        pdf.ln(5)

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
    except Exception:
        return b""


def export_report_html(messages: list, title: str = "Analysis Report") -> bytes:
    """
    Generate a structured HTML report from the conversation.
    Includes: title, date, conversation (user questions + AI answers),
    embedded chart images, and code snippets.
    """
    import base64, datetime as _dt

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    user_turns  = [m for m in messages if m["role"] == "user"]
    asst_turns  = [m for m in messages if m["role"] == "assistant"]
    n_charts    = sum(len(m.get("charts", [])) for m in asst_turns)
    n_code      = sum(len(m.get("code_snippets", [])) for m in asst_turns)

    def _img_tag(path: str) -> str:
        p = Path(path)
        if not p.exists():
            return ""
        data = base64.b64encode(p.read_bytes()).decode()
        return (f'<img src="data:image/png;base64,{data}" '
                f'style="max-width:100%;border-radius:8px;margin:8px 0;">')

    def _escape(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;")
                 .replace(">", "&gt;").replace("\n", "<br>"))

    rows_html = []
    pairs = list(zip(user_turns, asst_turns))
    if len(user_turns) > len(asst_turns):
        pairs.append((user_turns[-1], None))

    for i, (u, a) in enumerate(pairs, 1):
        charts_html = ""
        code_html   = ""
        if a:
            for cp in a.get("charts", []):
                charts_html += _img_tag(cp)
            for _s in a.get("code_snippets", []):
                _snip_code = _s["code"] if isinstance(_s, dict) else _s
                _snip_lbl  = (
                    "🎨 קוד גרף" if isinstance(_s, dict) and _s.get("type") == "chart"
                    else "📊 קוד ניתוח"
                )
                charts_html += (
                    f'<details style="margin:8px 0;">'
                    f'<summary style="cursor:pointer;color:#555;font-size:.85em;">{_snip_lbl}</summary>'
                    f'<pre style="background:#f4f4f4;padding:10px;border-radius:6px;'
                    f'font-size:.82em;overflow-x:auto;">{_escape(_snip_code)}</pre>'
                    f'</details>'
                )
        rows_html.append(f"""
        <div style="margin-bottom:28px;border-bottom:1px solid #e0e0e0;padding-bottom:20px;">
          <div style="margin-bottom:8px;">
            <span style="background:#1a73e8;color:white;padding:2px 10px;border-radius:12px;
                         font-size:.8em;font-weight:600;">שאלה {i}</span>
            <span style="font-size:.85em;color:#888;margin-right:8px;">{u['content'][:120]}</span>
          </div>
          <div style="background:#f8f9fa;border-radius:8px;padding:14px 18px;margin-bottom:6px;
                      line-height:1.7;color:#222;">{_escape(a['content'] if a else '')}</div>
          {charts_html}
        </div>""")

    html = f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body{{font-family:'Segoe UI',Arial,sans-serif;max-width:860px;margin:40px auto;
         padding:0 20px;color:#222;background:#fff;direction:rtl;}}
    h1{{color:#1a73e8;border-bottom:3px solid #1a73e8;padding-bottom:10px;}}
    .meta{{color:#888;font-size:.9em;margin-bottom:32px;}}
    .stat{{display:inline-block;background:#e8f0fe;color:#1a73e8;padding:4px 14px;
           border-radius:20px;font-size:.85em;font-weight:600;margin-left:8px;}}
  </style>
</head>
<body>
  <h1>📊 {title}</h1>
  <div class="meta">
    נוצר: {now}
    &nbsp;·&nbsp;
    <span class="stat">💬 {len(user_turns)} שאלות</span>
    <span class="stat">📈 {n_charts} גרפים</span>
    <span class="stat">🔎 {n_code} קטעי קוד</span>
  </div>
  {''.join(rows_html)}
  <p style="color:#aaa;font-size:.8em;text-align:center;margin-top:40px;">
    נוצר על ידי Data Analyst Chatbot · {now}
  </p>
</body>
</html>"""
    return html.encode("utf-8")


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
def inject_css(lang: str, theme: str = "dark") -> None:
    # Direction is always LTR to avoid bidi scrambling of mixed Hebrew+English.
    # For Hebrew we right-align paragraphs so the text reads naturally from right.
    direction = "ltr"
    text_align = "right" if lang == "he" else "left"
    tc = THEME_COLORS[theme]
    st.markdown(f"""
<style>
/* ═══════════════════════════════════════════════
   BASE STYLES  (mobile-first)
═══════════════════════════════════════════════ */
*, *::before, *::after {{ box-sizing: border-box; }}

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
body {{
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
    -webkit-text-size-adjust: 100%;   /* prevent iOS font inflation */
}}
.stApp, .stMarkdown {{
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
}}

/* ── Theme: override Streamlit chrome ── */
.stApp {{
    background-color: {tc['bg-app']} !important;
    color: {tc['text-primary']} !important;
}}
/* Force theme text color on ALL Streamlit native elements */
.stApp p, .stApp span, .stApp label, .stApp div,
.stApp li, .stApp td, .stApp th, .stApp caption,
.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li,
.stChatMessage, .stChatMessage p, .stChatMessage span,
.stChatMessage li, .stChatMessage td, .stChatMessage th,
[data-testid="stCaptionContainer"],
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stText"],
.uploadedFileName, .stFileUploader label,
.stFileUploader section, .stFileUploader div,
.stRadio label, .stCheckbox label,
.stMultiSelect span, .stNumberInput label,
.stSlider label, .stColorPicker label {{
    color: {tc['text-primary']} !important;
}}
/* Subtle text elements — secondary color */
[data-testid="stCaptionContainer"] {{
    color: {tc['text-secondary']} !important;
}}
.stFileUploader section {{
    background-color: {tc['bg-surface']} !important;
    border-color: {tc['border-default']} !important;
    color: {tc['text-secondary']} !important;
}}
.stFileUploader [data-testid="stFileUploaderDropzone"] {{
    background-color: {tc['bg-surface']} !important;
    border-color: {tc['border-strong']} !important;
}}
.stFileUploader [data-testid="stFileUploaderDropzone"] span,
.stFileUploader [data-testid="stFileUploaderDropzone"] small,
.stFileUploader [data-testid="stFileUploaderDropzone"] div {{
    color: {tc['text-secondary']} !important;
}}
/* Streamlit popover/dropdown menus */
[data-baseweb="popover"], [data-baseweb="menu"],
[data-baseweb="select"] [data-baseweb="menu"] {{
    background-color: {tc['bg-surface']} !important;
}}
[data-baseweb="menu"] li, [data-baseweb="menu"] li span {{
    color: {tc['text-primary']} !important;
}}
[data-baseweb="menu"] li:hover {{
    background-color: {tc['bg-hover']} !important;
}}
/* Bold / strong text */
.stApp strong, .stApp b, .stMarkdown strong {{
    color: {tc['text-primary']} !important;
}}
/* Dataframe / table styling */
.stDataFrame, .stTable {{
    color: {tc['text-primary']} !important;
}}
/* Streamlit alerts (info, warning, success, error) text */
[data-testid="stAlert"] p, [data-testid="stAlert"] span {{
    color: {tc['text-primary']} !important;
}}
/* Selectbox dropdown list text */
[data-baseweb="select"] span {{
    color: {tc['text-primary']} !important;
}}
/* Streamlit main block/header area */
[data-testid="stHeader"] {{
    background-color: {tc['bg-app']} !important;
}}
/* Separator/divider */
hr, .stApp hr {{
    border-color: {tc['border-default']} !important;
}}

/* ── LTR direction, right-aligned for Hebrew paragraphs.
   unicode-bidi:embed keeps Hebrew characters in their natural
   right-to-left order without flipping the whole container. ── */
.stChatMessage p, .stChatMessage li,
[data-testid="stInfo"] p,
[data-testid="stWarning"] p,
[data-testid="stSuccess"] p,
[data-testid="stError"] p,
[data-testid="stMarkdown"] p,
[data-testid="stMarkdown"] li {{
    direction: ltr;
    text-align: {text_align};
    unicode-bidi: embed;
}}

/* ── Header ── */
.app-header {{
    padding: 0.75rem 0 0.5rem;
    border-bottom: 1px solid {tc['border-default']};
    margin-bottom: 0.75rem;
}}
.app-title {{
    font-size: 1.5rem;        /* mobile default */
    font-weight: 700;
    color: {tc['text-primary']} !important;
    direction: {direction};
    text-align: {text_align};
    line-height: 1.2;
}}
.app-sub {{
    color: {tc['text-secondary']} !important;
    font-size: 0.85rem;
    direction: {direction};
    text-align: {text_align};
}}

/* ── Sidebar sections ── */
.sidebar-section {{
    background: {tc['bg-surface']};
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.8rem;
    border: 1px solid {tc['border-default']};
}}
.sidebar-label {{
    font-size: 0.78rem;
    font-weight: 600;
    color: {tc['text-secondary']} !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
    direction: {direction};
    text-align: {text_align};
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
    background: {tc['bg-surface']};
    border-radius: 8px;
    padding: 0.5rem 0.7rem;
    border: 1px solid {tc['border-default']};
    text-align: center;
}}
.metric-value {{
    font-size: 1.2rem;
    font-weight: 700;
    color: {tc['accent']} !important;
}}
.metric-label {{
    font-size: 0.7rem;
    color: {tc['text-secondary']} !important;
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
    border: 1px solid {tc['border-strong']};
    background: {tc['bg-surface']};
    color: {tc['text-primary']};
    transition: background 0.15s, color 0.15s, border-color 0.15s;
    cursor: pointer;
    -webkit-tap-highlight-color: transparent;
}}
.stButton > button:hover,
.stButton > button:focus-visible {{
    background: {tc['accent']};
    color: white;
    border-color: {tc['accent']};
    outline: none;
}}

/* ── Download buttons — distinct accent ── */
.stDownloadButton > button {{
    background: {tc['bg-surface']};
    color: {tc['accent']};
    border-color: {tc['border-strong']};
    font-weight: 600;
}}
.stDownloadButton > button:hover,
.stDownloadButton > button:focus-visible {{
    background: {tc['accent']};
    color: white;
    border-color: {tc['accent']};
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
    background: {tc['accent-bg']};
    color: {tc['accent']};
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
    border: 1px solid {tc['border-default']};
    margin-top: 0.5rem;
    max-width: 100%;
}}

/* ── Welcome card ── */
.welcome-card {{
    background: {tc['welcome-gradient']};
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border: 1px solid {tc['border-default']};
    direction: {direction};
}}

/* ── Quick-question chips row ── */
.chips-label {{
    font-size: .82em;
    color: {tc['text-secondary']} !important;
    margin: 14px 0 6px;
    font-weight: 500;
}}
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div[data-testid="stButton"] > button.chip-btn {{
    background: {tc['bg-surface']};
    color: {tc['accent']};
    border: 1px solid {tc['border-strong']};
    border-radius: 20px;
    font-size: .62em;
    padding: 3px 14px;
    min-height: 32px;
}}
div[data-testid="stButton"].chip-wrap button {{
    background: {tc['bg-surface']} !important;
    color: {tc['accent']} !important;
    border: 1px solid {tc['border-strong']} !important;
    border-radius: 20px !important;
    font-size: .62em !important;
    min-height: 32px !important;
    white-space: nowrap;
}}

/* ── Chart container entry animation ── */
[data-testid="stPlotlyChart"] {{
    animation: chartFadeIn 0.4s ease-out;
}}
@keyframes chartFadeIn {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

/* ── Thinking dots (typing indicator before streaming) ── */
.thinking-dots {{
    display: flex;
    gap: 6px;
    align-items: center;
    padding: 6px 2px 10px;
}}
.thinking-dots span {{
    width: 9px;
    height: 9px;
    background: {tc['accent']};
    border-radius: 50%;
    display: inline-block;
    animation: thinking-bounce 1.3s infinite ease-in-out;
    opacity: 0.5;
}}
.thinking-dots span:nth-child(2) {{ animation-delay: 0.22s; }}
.thinking-dots span:nth-child(3) {{ animation-delay: 0.44s; }}
@keyframes thinking-bounce {{
    0%, 60%, 100% {{ transform: translateY(0);    opacity: 0.4; }}
    30%            {{ transform: translateY(-7px); opacity: 1;   }}
}}

/* ── Dashboard guided empty state ── */
.dash-empty-state {{
    text-align: center;
    padding: 52px 24px 40px;
    color: {tc['text-secondary']};
}}
.dash-empty-icon {{
    font-size: 3.2rem;
    margin-bottom: 12px;
    line-height: 1;
}}
.dash-empty-title {{
    font-size: 1.25rem;
    color: {tc['text-primary']};
    font-weight: 600;
    margin-bottom: 24px;
}}
.dash-empty-steps {{
    display: flex;
    flex-direction: column;
    gap: 14px;
    max-width: 400px;
    margin: 0 auto 20px;
    text-align: left;
}}
.dash-step {{
    display: flex;
    align-items: flex-start;
    gap: 12px;
    font-size: 0.93rem;
    color: {tc['text-secondary']};
    line-height: 1.4;
}}
.step-num {{
    background: {tc['accent-bg']};
    color: {tc['accent']};
    border-radius: 50%;
    width: 26px;
    height: 26px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-weight: 700;
    font-size: 0.82rem;
}}
.dash-empty-tip {{
    font-size: 0.83rem;
    color: {tc['text-secondary']};
    margin-top: 20px;
    opacity: 0.85;
}}

/* ── Action icon bar CSS is injected inline near the marker element ── */
/* Force LTR on all buttons so mixed Hebrew+English labels read left-to-right */
.stButton > button,
.stDownloadButton > button {{
    direction: ltr !important;
    text-align: center !important;
}}

/* ── Sidebar conversation list ── */
.conv-section-hdr {{
    font-size: 0.72rem;
    font-weight: 700;
    color: {tc['text-muted']};
    text-transform: uppercase;
    letter-spacing: .07em;
    padding: 4px 2px 6px;
}}
.conv-group-label {{
    font-size: 0.68rem;
    font-weight: 700;
    color: {tc['text-muted']};
    text-transform: uppercase;
    letter-spacing: .06em;
    padding: 8px 2px 2px;
    margin-top: 6px;
}}
/* Make conversation buttons look like plain list items */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div div[data-testid="stButton"] button {{
    background: transparent;
    border: none;
    color: {tc['text-primary']};
    font-size: 0.87rem;
    text-align: left;
    padding: 5px 8px;
    border-radius: 6px;
    border-left: 3px solid transparent;
    justify-content: flex-start;
    font-weight: 400;
}}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div div[data-testid="stButton"] button:hover {{
    background: {tc['bg-surface']};
    color: {tc['text-primary']};
}}
.conv-active-indicator {{
    height: 2px;
    background: {tc['accent']};
    border-radius: 2px;
    margin: -6px 0 4px 0;
    opacity: 0.6;
}}

/* ── API key status badge ── */
.api-badge-ok {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: {tc['status-ok-bg']};
    color: {tc['status-ok-text']};
    border: 1px solid {tc['status-ok-border']};
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
    background: {tc['status-err-bg']};
    color: {tc['status-err-text']};
    border: 1px solid {tc['status-err-border']};
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
    color: {tc['text-muted']};
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
    position: fixed;
    top: -50px;
    left: 0;
    background: {tc['accent']};
    color: white;
    padding: 8px 16px;
    border-radius: 0 0 4px 0;
    font-weight: 600;
    z-index: 9999;
    text-decoration: none;
    transition: top 0.15s ease;
}}
.skip-link:focus,
.skip-link:focus-visible {{
    top: 0;
    outline: 3px solid #fff !important;
    outline-offset: -3px !important;
}}

/* Visible focus ring on every interactive element */
*:focus-visible {{
    outline: 3px solid {tc['accent']} !important;
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

/* Sidebar caption text — theme-aware contrast */
.sidebar-label {{
    color: {tc['text-secondary']} !important;
}}
.metric-label {{
    color: {tc['text-secondary']} !important;
}}
.api-hint {{
    color: {tc['text-muted']} !important;
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
   CHATGPT-STYLE LAYOUT ENHANCEMENTS
═══════════════════════════════════════════════ */

/* ── Centered chat container ── */
.stMainBlockContainer {{
    max-width: 900px;
    margin: 0 auto;
    padding-left: 1rem;
    padding-right: 1rem;
}}

/* ── User message bubble (rounded bg) ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
    background: {tc['bg-surface']} !important;
    border-radius: 18px !important;
    padding: 10px 16px !important;
    border: none !important;
}}
/* ── Assistant message (transparent, clean) ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {{
    background: transparent !important;
    padding: 10px 0 !important;
    border: none !important;
}}

/* ── Chat input — Calcalist-style floating pill ── */
[data-testid="stBottom"] {{
    padding-top: 24px;
    background: {tc['input-gradient']} !important;
}}
.stChatInput {{
    background: {tc['input-bg']} !important;
    border: 1.5px solid {tc['input-border']} !important;
    border-radius: 28px !important;
    box-shadow: {tc['input-shadow']} !important;
    padding: 4px 6px !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
    outline: none !important;
}}
.stChatInput > div, .stChatInput [data-baseweb] {{
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    background: transparent !important;
}}
.stChatInput *:focus-visible {{
    outline: none !important;
}}
.stChatInput textarea {{
    background: transparent !important;
    border: none !important;
    border-radius: 24px !important;
    padding: 10px 56px 10px 16px !important;
    color: {tc['text-primary']} !important;
    font-size: 1rem;
    caret-color: {tc['accent']} !important;
}}
.stChatInput textarea::placeholder {{
    color: {tc['input-placeholder']} !important;
    opacity: 1 !important;
}}
.stChatInput:focus-within {{
    border-color: {tc['input-focus-border']} !important;
    box-shadow: {tc['input-focus-shadow']} !important;
    outline: none !important;
}}
.stChatInput button {{
    border-radius: 50% !important;
    background: {tc['accent']} !important;
    color: white !important;
    border: none !important;
    width: 36px !important;
    height: 36px !important;
    min-width: 36px !important;
    min-height: 36px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: background 0.2s ease, transform 0.15s ease !important;
}}
.stChatInput button:hover {{
    background: {tc['accent-hover']} !important;
    transform: scale(1.05) !important;
}}
.stChatInput button:active {{
    transform: scale(0.95) !important;
}}

/* ── Sidebar — theme background ── */
[data-testid="stSidebar"] {{
    background-color: {tc['bg-sidebar']} !important;
    border-right: 1px solid {tc['border-subtle']} !important;
}}
[data-testid="stSidebar"] .stMarkdown {{
    color: {tc['text-secondary']};
}}

/* ── Scrollbar — thin & subtle ── */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {tc['scrollbar-thumb']}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {tc['scrollbar-hover']}; }}

/* ── Code blocks ── */
.stCodeBlock {{
    background: {tc['bg-code']} !important;
    border: 1px solid {tc['border-default']} !important;
    border-radius: 8px !important;
}}

/* ── Tabs — subtle bottom border ── */
.stTabs [data-baseweb="tab-list"] {{
    border-bottom: 1px solid {tc['border-default']};
    gap: 0;
}}
.stTabs [data-baseweb="tab"] {{
    color: {tc['text-secondary']};
    padding: 8px 20px;
    font-weight: 500;
}}
.stTabs [aria-selected="true"] {{
    color: {tc['text-primary']} !important;
    border-bottom-color: {tc['accent']} !important;
    font-weight: 600;
}}
/* Override Streamlit's default red tab indicator */
[data-baseweb="tab-highlight"] {{
    background-color: {tc['accent']} !important;
}}

/* ── Expander — harmonize ── */
.streamlit-expanderHeader {{
    color: {tc['text-primary']} !important;
    font-weight: 500;
}}
details[data-testid="stExpander"] {{
    border-color: {tc['border-default']} !important;
    background: {tc['bg-plot']} !important;
    border-radius: 8px !important;
}}

/* ── Select boxes / inputs ── */
.stSelectbox > div > div {{
    background: {tc['bg-surface']} !important;
    border-color: {tc['border-strong']} !important;
    color: {tc['text-primary']} !important;
}}
.stTextInput > div > div > input {{
    background: {tc['bg-surface']} !important;
    border-color: {tc['border-strong']} !important;
    color: {tc['text-primary']} !important;
    border-radius: 8px;
}}
.stTextInput > div > div > input:focus {{
    border-color: {tc['accent']} !important;
    box-shadow: 0 0 0 1px {tc['accent']}33 !important;
}}

/* ═══════════════════════════════════════════
   NATIVE COMPONENT OVERRIDES (light/dark)
═══════════════════════════════════════════ */

/* DataFrame & DataEditor — internal grid backgrounds */
[data-testid="stDataFrame"] > div,
[data-testid="stDataFrame"] [data-testid="glideDataEditor"],
[data-testid="stDataFrameResizable"],
.dvn-scroller {{
    background-color: {tc['bg-surface']} !important;
}}

/* Checkbox and radio — container + label */
.stCheckbox > label,
.stRadio > div {{
    color: {tc['text-primary']} !important;
}}
[data-testid="stCheckbox"] span {{
    color: {tc['text-primary']} !important;
}}

/* Multiselect — tag chips and dropdown */
[data-baseweb="tag"] {{
    background-color: {tc['bg-surface-raised']} !important;
    color: {tc['text-primary']} !important;
    border-color: {tc['border-default']} !important;
}}
[data-baseweb="select"] > div {{
    background-color: {tc['bg-surface']} !important;
    border-color: {tc['border-strong']} !important;
}}
[data-baseweb="input"] {{
    background-color: {tc['bg-surface']} !important;
}}

/* Number input */
.stNumberInput > div > div > input {{
    background: {tc['bg-surface']} !important;
    border-color: {tc['border-strong']} !important;
    color: {tc['text-primary']} !important;
    border-radius: 8px;
}}

/* Sidebar — full background + all children */
[data-testid="stSidebar"] > div:first-child {{
    background-color: {tc['bg-sidebar']} !important;
}}
[data-testid="stSidebar"] button {{
    color: {tc['text-primary']} !important;
}}

/* Primary button styling */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {{
    background: {tc['accent']} !important;
    color: white !important;
    border-color: {tc['accent']} !important;
}}

/* Alert boxes */
[data-testid="stAlert"] {{
    background-color: {tc['bg-surface']} !important;
    border-color: {tc['border-default']} !important;
}}

/* Tooltip and popover */
[data-baseweb="tooltip"] {{
    background-color: {tc['bg-surface-raised']} !important;
    color: {tc['text-primary']} !important;
}}

/* Main block container */
.main .block-container {{
    background-color: {tc['bg-app']} !important;
}}

/* Top toolbar area */
[data-testid="stToolbar"] {{
    background-color: {tc['bg-app']} !important;
}}
[data-testid="stDecoration"] {{
    background-image: none !important;
}}

/* Bottom container background (chat input area) */
[data-testid="stBottomBlockContainer"] {{
    background-color: {tc['bg-app']} !important;
}}

</style>
""", unsafe_allow_html=True)


# ─── Session State Init ────────────────────────────────────────────────────────
def init_state() -> None:
    if "lang" not in st.session_state:
        st.session_state.lang = "en"
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
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
    # Provider: "anthropic" | "openai" | "google" | "groq" | "ollama"
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
    # New provider API keys + model selectors (OpenAI, Google, Groq)
    for _pid, _pcfg in PROVIDER_REGISTRY.items():
        _sk = _pcfg["key_session"]
        if _sk and _pid != "anthropic" and _pid != "ollama":
            if _sk not in st.session_state:
                _ev = os.environ.get(_pcfg["env_var"], "") if _pcfg["env_var"] else ""
                try:
                    _sv = st.secrets.get(_pcfg["env_var"], "") if (not _ev and _pcfg["env_var"]) else ""
                except Exception:
                    _sv = ""
                st.session_state[_sk] = _ev or _sv
                st.session_state[f"{_sk}_from_env"] = bool(_ev or _sv)
            if f"{_sk}_from_env" not in st.session_state:
                st.session_state[f"{_sk}_from_env"] = False
        _mk = _pcfg["model_session"]
        if _mk and _mk != "ollama_model" and _mk not in st.session_state:
            st.session_state[_mk] = _pcfg["default_model"]
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
    # UX: hide quick-chips after user clicks one; reset after each AI reply
    if "chips_hidden" not in st.session_state:
        st.session_state.chips_hidden = False
    # Chat-based onboarding: tracks whether provider was explicitly chosen
    if "_provider_chosen" not in st.session_state:
        st.session_state._provider_chosen = False
    # Code panel visibility toggle (icon bar)
    if "show_code_panels" not in st.session_state:
        st.session_state.show_code_panels = True
    # Conversation persistence
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None   # None until first user message
    if "chat_just_loaded" not in st.session_state:
        st.session_state.chat_just_loaded = False
    # Track which language the dataset summary was generated in, and cache df for regen
    if "_summary_lang" not in st.session_state:
        st.session_state._summary_lang = None
    if "_summary_df" not in st.session_state:
        st.session_state._summary_df = None
    if "_summary_name" not in st.session_state:
        st.session_state._summary_name = None
    # AI Dashboard tab
    if "ai_dash_messages" not in st.session_state:
        st.session_state.ai_dash_messages = []
    if "ai_dash_chat" not in st.session_state:
        st.session_state.ai_dash_chat = None
    if "ai_dashboard_charts" not in st.session_state:
        st.session_state.ai_dashboard_charts = []
    # Streaming control (stop button)
    if "_is_streaming" not in st.session_state:
        st.session_state._is_streaming = False
    if "_partial_response" not in st.session_state:
        st.session_state._partial_response = ""
    if "_streaming_target" not in st.session_state:
        st.session_state._streaming_target = None  # "chat" or "ai_dash"


def get_active_key() -> str:
    """Return the API key for the currently selected provider."""
    provider = st.session_state.get("provider", "anthropic")
    cfg = PROVIDER_REGISTRY.get(provider)
    if cfg and cfg["key_session"]:
        return st.session_state.get(cfg["key_session"], "")
    return ""


def build_chat():
    """Build a chatlas chat object for the active provider."""
    provider = st.session_state.get("provider", "anthropic")
    cfg = PROVIDER_REGISTRY[provider]

    if provider == "ollama":
        model_name = st.session_state.get("ollama_model", "")
        if not model_name:
            raise ValueError(
                "לא נבחר מודל Ollama — בחר מודל מהתפריט בסייד-בר לאחר הרצת: ollama pull llama3.2\n"
                "No Ollama model selected — run: ollama pull llama3.2, then choose from the sidebar."
            )
        from chatlas import ChatOllama
        chat = ChatOllama(model=model_name, system_prompt=SYSTEM_PROMPT)
    else:
        key = get_active_key()
        model = (st.session_state.get(cfg["model_session"], cfg["default_model"])
                 if cfg["model_session"] else cfg["default_model"])

        _CHAT_CLASSES = {
            "anthropic": ChatAnthropic,
            "openai":    ChatOpenAI,
            "google":    ChatGoogle,
            "groq":      ChatGroq,
        }
        ChatClass = _CHAT_CLASSES[provider]

        kwargs = dict(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            api_key=key if key else None,
        )
        if cfg["max_tokens"]:
            kwargs["max_tokens"] = cfg["max_tokens"]

        chat = ChatClass(**kwargs)

    chat.register_tool(tools.get_data_overview)
    chat.register_tool(tools.run_analysis)
    chat.register_tool(tools.create_chart)
    chat.register_tool(tools.suggest_next_analyses)
    return chat


def build_ai_dash_chat():
    """Build a chatlas chat object for the AI Dashboard tab (dashboard-specific tools)."""
    provider = st.session_state.get("provider", "anthropic")
    cfg = PROVIDER_REGISTRY[provider]

    if provider == "ollama":
        model_name = st.session_state.get("ollama_model", "")
        if not model_name:
            raise ValueError(
                "לא נבחר מודל Ollama — בחר מודל מהתפריט בסייד-בר\n"
                "No Ollama model selected — choose from the sidebar."
            )
        from chatlas import ChatOllama
        chat = ChatOllama(model=model_name, system_prompt=AI_DASHBOARD_SYSTEM_PROMPT)
    else:
        key = get_active_key()
        model = (st.session_state.get(cfg["model_session"], cfg["default_model"])
                 if cfg["model_session"] else cfg["default_model"])
        _CHAT_CLASSES = {
            "anthropic": ChatAnthropic,
            "openai":    ChatOpenAI,
            "google":    ChatGoogle,
            "groq":      ChatGroq,
        }
        ChatClass = _CHAT_CLASSES[provider]
        kwargs = dict(
            model=model,
            system_prompt=AI_DASHBOARD_SYSTEM_PROMPT,
            api_key=key if key else None,
        )
        if cfg["max_tokens"]:
            kwargs["max_tokens"] = cfg["max_tokens"]
        chat = ChatClass(**kwargs)

    # Register dashboard-specific tools
    chat.register_tool(tools.get_data_overview)
    chat.register_tool(tools.set_dashboard_charts)
    chat.register_tool(tools.add_dashboard_chart)
    chat.register_tool(tools.update_dashboard_chart)
    chat.register_tool(tools.remove_dashboard_chart)
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


def _manual_stream(stream_generator, placeholder):
    """Stream chunks manually via st.empty(), saving partial results.

    Replaces st.write_stream() so we can save partial text to session state
    for recovery if the user stops the streaming mid-way.
    """
    full_parts: list[str] = []
    for chunk in stream_generator:
        full_parts.append(chunk)
        text_so_far = "".join(full_parts)
        st.session_state._partial_response = text_so_far
        placeholder.markdown(text_so_far + "▌")
    # Done — remove typing cursor
    final = "".join(full_parts)
    placeholder.markdown(final)
    return final


# ─── Provider Section ─────────────────────────────────────────────────────────

def _render_api_key_section(provider: str, T: dict) -> None:
    """Generic API key + model selector for any keyed provider."""
    cfg = PROVIDER_REGISTRY[provider]
    sess_key = cfg["key_session"]
    env_flag = f"{sess_key}_from_env"
    has_key = bool(st.session_state.get(sess_key, ""))

    if has_key:
        label = T["api_from_env"] if st.session_state.get(env_flag) else T["api_set"]
        st.markdown(f'<div class="api-badge-ok">{label}</div>', unsafe_allow_html=True)
        if not st.session_state.get(env_flag):
            if st.button(T["api_clear"], use_container_width=True,
                         key=f"api_clear_{provider}"):
                st.session_state[sess_key] = ""
                st.session_state.chat = None
                st.session_state.messages = []
                st.rerun()
    else:
        st.markdown(f'<div class="api-badge-err">{T["api_missing"]}</div>',
                    unsafe_allow_html=True)
        new_key = st.text_input(
            T.get(f"{provider}_model_lbl", T["api_label"]),
            type="password",
            placeholder=cfg["placeholder"],
            help=T["api_help"],
            label_visibility="collapsed",
            key=f"api_input_{provider}",
        )
        if st.button(T["api_save"], use_container_width=True,
                     key=f"api_save_{provider}"):
            cleaned = new_key.strip()
            if cleaned.startswith(cfg["key_prefix"]):
                st.session_state[sess_key] = cleaned
                st.session_state[env_flag] = False
                if st.session_state.data_loaded:
                    st.session_state.chat = build_chat()
                st.rerun()
            else:
                _prefix = cfg["key_prefix"]
                st.error(f"Key must start with {_prefix}  /  המפתח חייב להתחיל ב- {_prefix}")
        st.markdown(f'<div class="api-hint">🔗 {cfg["hint_url"]}</div>',
                    unsafe_allow_html=True)

    # Model selector (when provider has multiple models and key is set)
    if has_key and cfg["model_session"] and cfg["models"]:
        model_key = cfg["model_session"]
        cur_model = st.session_state.get(model_key, cfg["default_model"])
        idx = cfg["models"].index(cur_model) if cur_model in cfg["models"] else 0
        lbl_key = f"{provider}_model_lbl"
        chosen = st.selectbox(
            T.get(lbl_key, "Model"),
            cfg["models"],
            index=idx,
            key=f"model_select_{provider}",
            label_visibility="collapsed",
        )
        if chosen != st.session_state.get(model_key):
            st.session_state[model_key] = chosen
            if st.session_state.data_loaded:
                st.session_state.chat = build_chat()


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
            st.rerun()  # advance onboarding automatically

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
    """Provider selector: 5 providers via selectbox."""
    st.markdown(f'<div class="sidebar-label">{T["provider_header"]}</div>',
                unsafe_allow_html=True)

    provider_ids = list(PROVIDER_REGISTRY.keys())
    cur_provider = st.session_state.get("provider", "anthropic")
    cur_idx = provider_ids.index(cur_provider) if cur_provider in provider_ids else 0

    provider = st.selectbox(
        T["provider_header"],
        options=provider_ids,
        format_func=lambda x: T[PROVIDER_REGISTRY[x]["label_key"]],
        index=cur_idx,
        label_visibility="collapsed",
        key="provider_select",
    )
    if provider != cur_provider:
        st.session_state.provider = provider
        st.session_state.chat = None
        st.rerun()

    st.markdown('<div style="margin-top:0.4rem"></div>', unsafe_allow_html=True)
    if provider == "ollama":
        _render_ollama_section(T)
    else:
        _render_api_key_section(provider, T)


# ─── Upload helpers (shared by inline + sidebar) ──────────────────────────────
_MAX_ROWS       = 100_000
_MEMORY_WARN_MB = 400


def _make_dataset_summary(df, name: str, T: dict) -> str:
    """Return a markdown string summarising the newly loaded dataset."""
    num_cols  = len(df.select_dtypes(include="number").columns)
    cat_cols  = len(df.select_dtypes(include="object").columns)
    date_cols = len(df.select_dtypes(include="datetime").columns)
    missing   = int(df.isnull().sum().sum())
    miss_str  = (T["ds_missing_ok"] if missing == 0
                 else T["ds_missing_warn"].format(n=missing))
    col_list  = ", ".join(f"`{c}`" for c in df.columns[:7])
    if len(df.columns) > 7:
        col_list += f" +{len(df.columns) - 7}"
    return (
        f"📊 **{name}** {T['ds_loaded']}\n\n"
        f"| | |\n|---|---|\n"
        f"| **{T['ds_rows']}** | {len(df):,} |\n"
        f"| **{T['ds_cols']}** | {len(df.columns)} |\n"
        f"| **{T['ds_numeric']}** | {num_cols} |\n"
        f"| **{T['ds_categorical']}** | {cat_cols} |\n"
        f"| **{T['ds_dates']}** | {date_cols} |\n"
        f"| **{T['ds_data']}** | {miss_str} |\n\n"
        f"**{T['ds_col_label']}** {col_list}\n\n"
        f"{T['ds_prompt']}"
    )


def _init_dataset(df, name: str, file_id, T: dict) -> None:
    """Shared post-load state initialisation (used by both upload paths)."""
    tools.set_dataframe(df, name=name)
    st.session_state.data_loaded        = True
    st.session_state._uploaded_file_id  = file_id
    st.session_state.chat               = None
    st.session_state.dashboard_charts   = []
    st.session_state.data_warnings      = validate_dataframe(df)
    # Clear AI Dashboard state
    st.session_state.ai_dashboard_charts = []
    st.session_state.ai_dash_messages    = []
    st.session_state.ai_dash_chat        = None
    tools.clear_ai_dashboard_charts()
    # First assistant message = dataset summary card
    st.session_state.messages = [{
        "role":          "assistant",
        "content":       _make_dataset_summary(df, name, T),
        "charts":        [],
        "chart_configs": [],
        "code_snippets": [],
        "is_system":     True,
    }]
    st.session_state._summary_lang = st.session_state.lang
    # Cache df + name for language-switch regeneration (tools._df is module-level, lost on reload)
    st.session_state._summary_df   = df
    st.session_state._summary_name = name


def _handle_uploaded_file(uploaded, T: dict) -> None:
    """Load a file widget result; skip if already processed."""
    file_id = f"{uploaded.name}_{uploaded.size}"
    if file_id == st.session_state.get("_uploaded_file_id"):
        return
    try:
        df = load_uploaded_file(uploaded)
        total_rows = len(df)
        if total_rows > _MAX_ROWS:
            df = df.sample(_MAX_ROWS, random_state=42).reset_index(drop=True)
            st.info(T["large_file_notice"].format(n=_MAX_ROWS, total=total_rows))
        mb = check_df_memory(df)
        if mb > _MEMORY_WARN_MB:
            st.warning(T["memory_warn"].format(mb=int(mb)))
        sensitive = detect_sensitive_columns(df)
        if sensitive:
            st.warning(T["bias_warning"].format(cols=", ".join(sensitive)))
        _init_dataset(df, uploaded.name, file_id, T)
        _audit("FILE_UPLOAD",
               f"name={uploaded.name} rows={len(df)} cols={len(df.columns)} mb={mb:.1f}")
        st.rerun()
    except Exception as exc:
        _audit("FILE_UPLOAD_ERROR", str(exc))
        st.error(str(exc))


def _handle_demo_data(T: dict) -> None:
    """Load sample data and initialise session."""
    df = make_sample_df()
    _init_dataset(df, "sample_sales.csv", "__demo__", T)
    _audit("DEMO_LOADED", "")
    st.rerun()


def _render_inline_upload(T: dict) -> None:
    """Full-page upload prompt shown inside the chat tab when no data is loaded."""
    st.markdown(
        f'<div class="upload-zone">'
        f'<h2>{T["upload_inline_title"]}</h2>'
        f'<p>{T["upload_inline_sub"]}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        T["upload_label"],
        type=["csv", "tsv", "xlsx", "xls", "json", "parquet"],
        label_visibility="collapsed",
        help=T["upload_help"],
        key="inline_uploader",
    )
    if uploaded:
        _handle_uploaded_file(uploaded, T)

    st.markdown(
        f'<div class="upload-divider">{T["upload_inline_or"]}</div>',
        unsafe_allow_html=True,
    )
    if st.button(T["upload_inline_demo"], use_container_width=False,
                 key="demo_btn_inline"):
        _handle_demo_data(T)


# ─── Chat-Based Onboarding ──────────────────────────────────────────────────

def _get_onboarding_step() -> str:
    """Determine current onboarding step based on session state.

    Returns one of: 'welcome', 'api_key', 'data', 'done'.
    """
    provider = st.session_state.get("provider", "anthropic")
    if provider == "ollama":
        has_key = (st.session_state.get("ollama_available", False)
                   and bool(st.session_state.get("ollama_model")))
    else:
        has_key = bool(get_active_key())
    has_data = st.session_state.get("data_loaded", False)

    if not has_key:
        # Check if provider has been explicitly chosen by the user
        if st.session_state.get("_provider_chosen"):
            return "api_key"
        return "welcome"
    if not has_data:
        return "data"
    return "done"


def _render_onboarding_chat(T: dict) -> None:
    """Render onboarding flow as chat messages inside the chat tab.

    Replaces the old upload-zone widget and tour bar with a natural
    conversational flow: welcome → provider → API key → data upload.
    """
    step = _get_onboarding_step()

    # ── Step 1: Welcome — provider selection (5 buttons: 3+2 grid) ──────
    if step == "welcome":
        with st.chat_message("assistant"):
            st.markdown(T["onboard_welcome"])
            # Row 1: cloud providers
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button(T["provider_cloud"], key="onboard_anthropic",
                             use_container_width=True):
                    st.session_state.provider = "anthropic"
                    st.session_state._provider_chosen = True
                    st.rerun()
            with c2:
                if st.button(T["provider_openai"], key="onboard_openai",
                             use_container_width=True):
                    st.session_state.provider = "openai"
                    st.session_state._provider_chosen = True
                    st.rerun()
            with c3:
                if st.button(T["provider_google"], key="onboard_google",
                             use_container_width=True):
                    st.session_state.provider = "google"
                    st.session_state._provider_chosen = True
                    st.rerun()
            # Row 2: speed + local
            c4, c5 = st.columns(2)
            with c4:
                if st.button(T["provider_groq"], key="onboard_groq",
                             use_container_width=True):
                    st.session_state.provider = "groq"
                    st.session_state._provider_chosen = True
                    st.rerun()
            with c5:
                if st.button(T["provider_local"], key="onboard_ollama",
                             use_container_width=True):
                    st.session_state.provider = "ollama"
                    st.session_state._provider_chosen = True
                    st.rerun()

    # ── Step 2: API key / Ollama setup (generic) ──────────────────────────
    elif step == "api_key":
        provider = st.session_state.get("provider", "anthropic")
        cfg = PROVIDER_REGISTRY[provider]
        provider_label = T[cfg["label_key"]]
        # Show chosen provider as context
        with st.chat_message("assistant"):
            st.markdown(f"✅ {provider_label}")

        if provider == "ollama":
            with st.chat_message("assistant"):
                st.markdown(T["onboard_ollama_prompt"])
                _render_ollama_section(T)
        else:
            prompt_key = f"onboard_{provider}_prompt"
            prompt_text = T.get(prompt_key, T["onboard_api_key_prompt"])
            with st.chat_message("assistant"):
                st.markdown(prompt_text)
                new_key = st.text_input(
                    T.get(f"{provider}_model_lbl", T["api_label"]),
                    type="password",
                    placeholder=cfg["placeholder"],
                    label_visibility="collapsed",
                    key="onboard_api_input",
                )
                if st.button(T["api_save"], key="onboard_api_save",
                             use_container_width=False):
                    cleaned = new_key.strip()
                    if cleaned.startswith(cfg["key_prefix"]):
                        st.session_state[cfg["key_session"]] = cleaned
                        st.session_state[f"{cfg['key_session']}_from_env"] = False
                        st.rerun()
                    else:
                        _prefix = cfg["key_prefix"]
                        st.error(f"Key must start with {_prefix}  /  המפתח חייב להתחיל ב- {_prefix}")
                st.caption(f"🔗 {cfg['hint_url']}")

    # ── Step 3: Data upload ───────────────────────────────────────────────
    elif step == "data":
        with st.chat_message("assistant"):
            st.markdown(T["onboard_data_prompt"])

            uploaded = st.file_uploader(
                T["upload_label"],
                type=["csv", "tsv", "xlsx", "xls", "json", "parquet"],
                label_visibility="collapsed",
                help=T["upload_help"],
                key="onboard_uploader",
            )
            if uploaded:
                _handle_uploaded_file(uploaded, T)

            st.markdown(f"**{T['upload_inline_or']}**")
            if st.button(
                T["upload_inline_demo"],
                key="onboard_demo_btn",
                use_container_width=False,
            ):
                _handle_demo_data(T)


def _build_context_chips(T: dict) -> list[str]:
    """Generate context-aware analysis suggestions based on the loaded dataset."""
    df = tools.get_dataframe()
    if df is None:
        return T.get("quick_qs", [])[:5]

    lang = st.session_state.lang
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()
    suggestions: list[str] = []

    if lang == "he":
        suggestions.append("תן לי סקירה כללית של הנתונים")
        if len(num_cols) >= 2:
            suggestions.append(f"מה הקורלציה בין {num_cols[0]} ל-{num_cols[1]}?")
        if cat_cols and num_cols:
            suggestions.append(f"השווה {num_cols[0]} לפי {cat_cols[0]}")
        if date_cols and num_cols:
            suggestions.append(f"הצג מגמת {num_cols[0]} לאורך זמן")
        elif num_cols:
            suggestions.append(f"מה ההתפלגות של {num_cols[0]}?")
        if cat_cols:
            suggestions.append(f"מה הערכים הנפוצים ב-{cat_cols[0]}?")
        if len(suggestions) < 4 and num_cols:
            suggestions.append("הצג גרף עם התובנות המעניינות ביותר")
    else:
        suggestions.append("Give me an overview of the data")
        if len(num_cols) >= 2:
            suggestions.append(f"Correlation between {num_cols[0]} and {num_cols[1]}?")
        if cat_cols and num_cols:
            suggestions.append(f"Compare {num_cols[0]} by {cat_cols[0]}")
        if date_cols and num_cols:
            suggestions.append(f"Show {num_cols[0]} trend over time")
        elif num_cols:
            suggestions.append(f"Distribution of {num_cols[0]}?")
        if cat_cols:
            suggestions.append(f"Most common values in {cat_cols[0]}?")
        if len(suggestions) < 4 and num_cols:
            suggestions.append("Show a chart with the most interesting insights")

    return suggestions[:5]


def _render_quick_chips(T: dict, context_aware: bool = False) -> None:
    """Row of quick-question chip buttons shown after the last AI reply.
    Auto-dismisses when the user clicks one (chips_hidden flag)."""
    # Dismissed after a chip click; re-shown after next AI reply
    if st.session_state.get("chips_hidden"):
        return
    qs = _build_context_chips(T) if context_aware else T.get("quick_qs", [])[:5]
    if not qs:
        return
    st.markdown(
        f'<div class="chips-label">{T["quick_chips_lbl"]}</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(min(len(qs), 3))
    for i, q in enumerate(qs):
        with cols[i % len(cols)]:
            if st.button(q, key=f"chip_q_{i}", use_container_width=True):
                st.session_state.pending_input = q
                st.session_state.chips_hidden = True   # ← auto-dismiss
                st.rerun()


# ─── Conversation Persistence ─────────────────────────────────────────────────

_CHATS_DIR = Path(__file__).parent / "chats"
_CHATS_DIR.mkdir(exist_ok=True)


def _chat_path(chat_id: str) -> Path:
    return _CHATS_DIR / f"{chat_id}.json"


def _new_chat_id() -> str:
    import uuid as _uuid
    ts = int(datetime.datetime.now().timestamp())
    uid = _uuid.uuid4().hex[:8]
    return f"{ts}_{uid}"


def _save_chat(chat_id: str, messages: list, name: str) -> None:
    """Persist current conversation to disk."""
    import json as _json
    p = _chat_path(chat_id)
    existing: dict = {}
    if p.exists():
        try:
            existing = _json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    data = {
        "id":         chat_id,
        "name":       name[:60],
        "messages":   messages,
        "created_at": existing.get(
            "created_at",
            datetime.datetime.now().isoformat(timespec="seconds"),
        ),
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    p.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _list_chats() -> list:
    """Return all saved chats sorted by updated_at descending."""
    import json as _json
    chats = []
    for f in sorted(
        _CHATS_DIR.glob("*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    ):
        try:
            d = _json.loads(f.read_text(encoding="utf-8"))
            chats.append(d)
        except Exception:
            pass
    return chats


def _load_chat(chat_id: str):
    """Load a saved chat from disk. Returns None if not found."""
    import json as _json
    p = _chat_path(chat_id)
    if not p.exists():
        return None
    try:
        return _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _delete_chat(chat_id: str) -> None:
    p = _chat_path(chat_id)
    if p.exists():
        p.unlink()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar(T: dict) -> None:
    with st.sidebar:
        # ── "New Conversation" button ──────────────────────────────────────
        if st.button(
            T["new_conv_btn"],
            use_container_width=True,
            key="new_conv_btn_sidebar",
            type="primary",
        ):
            st.session_state.messages = []
            st.session_state.chat = None
            st.session_state.current_chat_id = None
            st.session_state.chips_hidden = False
            st.session_state._request_count = 0
            st.session_state._partial_response = ""
            st.session_state._is_streaming = False
            st.session_state._streaming_target = None
            st.rerun()

        st.markdown("---")

        # ── Conversations ─────────────────────────────────────────────────
        st.markdown(
            f'<div class="conv-section-hdr">{T["conv_history_hdr"]}</div>',
            unsafe_allow_html=True,
        )

        _saved_chats = _list_chats()
        if not _saved_chats:
            st.caption(T["no_saved_chats"])
        else:
            _today = datetime.date.today()
            _today_chats = []
            _earlier_chats = []
            for _c in _saved_chats:
                try:
                    _upd = datetime.datetime.fromisoformat(_c.get("updated_at", "")).date()
                    if _upd == _today:
                        _today_chats.append(_c)
                    else:
                        _earlier_chats.append(_c)
                except Exception:
                    _earlier_chats.append(_c)

            def _render_chat_list(chat_list: list, group_key: str) -> None:
                for _ci, _ch in enumerate(chat_list):
                    _cid  = _ch.get("id", "")
                    _name = _ch.get("name", _cid) or _cid
                    _is_active = _cid == st.session_state.current_chat_id
                    _active_cls = " active" if _is_active else ""

                    # Row: name button + delete button
                    _col_name, _col_del = st.columns([9, 1])
                    with _col_name:
                        _btn_label = _name[:38] + ("…" if len(_name) > 38 else "")
                        if st.button(
                            _btn_label,
                            key=f"conv_load_{group_key}_{_ci}",
                            use_container_width=True,
                            help=_name,
                        ):
                            _loaded = _load_chat(_cid)
                            if _loaded:
                                st.session_state.messages = _loaded.get("messages", [])
                                st.session_state.current_chat_id = _cid
                                st.session_state.chat = None   # fresh AI context
                                st.session_state.chat_just_loaded = True
                                st.session_state.chips_hidden = False
                                st.rerun()
                    with _col_del:
                        if st.button(
                            T["del_chat_btn"],
                            key=f"conv_del_{group_key}_{_ci}",
                            help=T["del_chat_help"],
                        ):
                            _delete_chat(_cid)
                            if st.session_state.current_chat_id == _cid:
                                st.session_state.messages = []
                                st.session_state.chat = None
                                st.session_state.current_chat_id = None
                            st.rerun()

                    # Highlight active chat with CSS class via custom HTML
                    if _is_active:
                        st.markdown(
                            '<div class="conv-active-indicator"></div>',
                            unsafe_allow_html=True,
                        )

            if _today_chats:
                st.markdown(
                    f'<div class="conv-group-label">{T["chat_date_today"]}</div>',
                    unsafe_allow_html=True,
                )
                _render_chat_list(_today_chats, "today")

            if _earlier_chats:
                st.markdown(
                    f'<div class="conv-group-label">{T["chat_date_earlier"]}</div>',
                    unsafe_allow_html=True,
                )
                _render_chat_list(_earlier_chats, "earlier")

        st.markdown("---")

        # ── Language toggle (standalone) ──────────────────────────────────
        if st.button(T["lang_btn"], use_container_width=True, key="lang_toggle"):
            st.session_state.lang = "en" if st.session_state.lang == "he" else "he"
            st.rerun()

        # ── Theme toggle ─────────────────────────────────────────────────
        _theme_label = T["theme_btn_light"] if st.session_state.theme == "dark" else T["theme_btn_dark"]
        if st.button(_theme_label, use_container_width=True, key="theme_toggle"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()


# ─── Action Icon Bar (export + code toggle) ──────────────────────────────────
def _render_action_icons(T: dict) -> None:
    """Compact icon bar: export downloads + code-panel toggle."""
    msgs = st.session_state.messages
    if not msgs:
        return
    _has_charts = any(msg.get("charts") for msg in msgs)
    _has_code = any(msg.get("code_snippets") for msg in msgs)
    _tc = THEME_COLORS[st.session_state.theme]
    with st.container():
        st.markdown(f"""<span id="action-icons-bar"></span>
<style>
div[data-testid="stLayoutWrapper"]:has(#action-icons-bar) div[data-testid="stVerticalBlock"] {{
    gap: 0 !important;
}}
div[data-testid="stLayoutWrapper"]:has(#action-icons-bar) .stDownloadButton button,
div[data-testid="stLayoutWrapper"]:has(#action-icons-bar) .stButton button {{
    padding: 6px 10px !important;
    min-height: unset !important;
    height: 36px !important;
    width: 36px !important;
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 1.15rem !important;
    color: {_tc['text-tertiary']} !important;
    box-shadow: none !important;
    outline: none !important;
    cursor: pointer !important;
}}
div[data-testid="stLayoutWrapper"]:has(#action-icons-bar) .stDownloadButton button:hover,
div[data-testid="stLayoutWrapper"]:has(#action-icons-bar) .stButton button:hover {{
    background-color: {_tc['bg-hover']} !important;
    color: {_tc['text-primary']} !important;
}}
div[data-testid="stLayoutWrapper"]:has(#action-icons-bar) .stDownloadButton svg {{
    display: none !important;
}}
div[data-testid="stLayoutWrapper"]:has(#action-icons-bar) .stDownloadButton button div {{
    gap: 0 !important;
}}
div[data-testid="stLayoutWrapper"]:has(#action-icons-bar) div[data-testid="stHorizontalBlock"] {{
    gap: 0.3rem !important;
}}
div[data-testid="stLayoutWrapper"]:has(#action-icons-bar) div[data-testid="stColumn"] {{
    padding: 0 !important; min-width: 0 !important; flex: 0 0 auto !important; width: auto !important;
}}
div[data-testid="stLayoutWrapper"]:has(#action-icons-bar) button:disabled {{
    color: {_tc['text-disabled']} !important; opacity: 0.4 !important;
}}
div[data-testid="stLayoutWrapper"]:has(#action-icons-bar) iframe {{
    border: none !important; background: transparent !important;
    height: 42px !important; width: 42px !important;
}}
</style>""", unsafe_allow_html=True)
        _ic = st.columns([1, 1, 1, 1, 1, 1, 15], gap="small")
        with _ic[0]:
            try:
                _d = export_chat_html(st.session_state.messages, T["page_title"]) or b""
            except Exception:
                _d = b""
            if _d:
                st.download_button("📄", data=_d, file_name="chat_export.html",
                                   mime="text/html", help=T["export_chat_help"],
                                   key="export_chat_btn")
            else:
                st.button("📄", disabled=True, help=T["export_chat_help"],
                          key="export_html_dis")
        with _ic[1]:
            try:
                _d = export_chat_pdf(st.session_state.messages, T["page_title"]) or b""
            except Exception:
                _d = b""
            if _d:
                st.download_button("📑", data=_d, file_name="chat_export.pdf",
                                   mime="application/pdf", help=T["export_pdf_help"],
                                   key="export_pdf_btn")
            else:
                st.button("📑", disabled=True, help=T["export_pdf_help"],
                          key="export_pdf_dis")
        with _ic[2]:
            try:
                _d = export_report_html(st.session_state.messages, T["page_title"]) or b""
            except Exception:
                _d = b""
            if _d:
                st.download_button("📊", data=_d, file_name="analysis_report.html",
                                   mime="text/html", help=T["export_report_help"],
                                   key="export_report_btn")
            else:
                st.button("📊", disabled=True, help=T["export_report_help"],
                          key="export_report_dis")
        with _ic[3]:
            try:
                _d = export_ai_charts_zip(st.session_state.messages) if _has_charts else b""
            except Exception:
                _d = b""
            if _d:
                st.download_button("🖼", data=_d, file_name="ai_charts.zip",
                                   mime="application/zip", help=T["export_ai_charts_help"],
                                   key="export_charts_btn")
            else:
                st.button("🖼", disabled=True, help=T["export_ai_charts_help"],
                          key="export_charts_dis")
        with _ic[4]:
            if _has_code:
                _code_icon = "💻" if st.session_state.show_code_panels else "💻"
                if st.button(_code_icon, help=T["code_toggle_help"],
                             key="toggle_code_btn"):
                    st.session_state.show_code_panels = not st.session_state.show_code_panels
                    st.rerun()
        with _ic[5]:
            # Copy last assistant response to clipboard via inline HTML component
            import json as _json
            _last_asst = next(
                (m["content"] for m in reversed(msgs)
                 if m["role"] == "assistant" and not m.get("is_system")),
                None,
            )
            if _last_asst:
                _escaped = _json.dumps(_last_asst)
                import streamlit.components.v1 as _components
                _components.html(f"""
<style>
*{{ margin:0; padding:0; }}
body{{ background:transparent; overflow:hidden; }}
button#cpbtn{{
    padding:6px 10px; height:36px; width:36px;
    background:transparent; border:none; border-radius:8px;
    font-size:1.15rem; color:{_tc['text-tertiary']}; cursor:pointer; line-height:1;
}}
button#cpbtn:hover{{ background:{_tc['bg-hover']}; color:{_tc['text-primary']}; }}
</style>
<button id="cpbtn" title="{T['copy_response_help']}" onclick="copyText()">📋</button>
<script>
function copyText(){{
  var txt={_escaped};
  if(navigator.clipboard && window.isSecureContext){{
    navigator.clipboard.writeText(txt).then(ok,fb);
  }} else {{ fb(); }}
  function fb(){{
    var ta=document.createElement('textarea');
    ta.value=txt; ta.style.position='fixed'; ta.style.opacity='0';
    document.body.appendChild(ta); ta.select();
    try{{ document.execCommand('copy'); ok(); }}catch(e){{}}
    document.body.removeChild(ta);
  }}
  function ok(){{
    var b=document.getElementById('cpbtn');
    b.textContent='✅';
    setTimeout(function(){{ b.textContent='📋'; }},1500);
  }}
}}
</script>
""", height=42)
            else:
                st.button("📋", disabled=True, help=T["copy_response_help"],
                          key="copy_resp_dis")


# ─── Chat History ─────────────────────────────────────────────────────────────
def render_history() -> None:
    T = TEXT[st.session_state.lang]
    msgs = st.session_state.messages
    # Index of the last real assistant message (not is_system, not user)
    last_asst_idx = max(
        (i for i, m in enumerate(msgs)
         if m["role"] == "assistant" and not m.get("is_system")),
        default=-1,
    )

    for msg_idx, msg in enumerate(msgs):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # ── Charts (with Add-to-Dashboard button) ────────────────────
            chart_configs = msg.get("chart_configs", [])
            for ci, chart_path in enumerate(msg.get("charts", [])):
                p = Path(chart_path)
                if p.exists():
                    st.image(str(p), use_container_width=True)
                    btn_key = f"add_dash_{msg_idx}_{ci}"
                    if st.button(
                        T["add_to_dash_from_chat"],
                        key=btn_key,
                        help=T.get("add_to_dash", ""),
                    ):
                        cfg = (chart_configs[ci]
                               if ci < len(chart_configs) else None)
                        if cfg:
                            st.session_state.dashboard_charts.append(cfg)
                        else:
                            st.session_state.dashboard_charts.append({
                                "type": "image",
                                "path": str(p),
                                "title": Path(p).stem,
                            })
                        st.toast(T["chart_added_from_chat"])

            # ── Code panel (analysis + chart tabs) ───────────────────────
            _snippets = msg.get("code_snippets", [])
            if _snippets and st.session_state.get("show_code_panels", True):
                # Normalise: support legacy str format and new dict format
                _analysis = [
                    (s["code"] if isinstance(s, dict) else s)
                    for s in _snippets
                    if not isinstance(s, dict) or s.get("type") == "analysis"
                ]
                _charts_code = [
                    s["code"]
                    for s in _snippets
                    if isinstance(s, dict) and s.get("type") == "chart"
                ]
                with st.expander(T["code_panel_lbl"], expanded=True):
                    st.caption(T["code_panel_hint"])
                    if _analysis and _charts_code:
                        # Two tabs: data query + chart code
                        _tab_a, _tab_c = st.tabs(
                            [T["code_tab_analysis"], T["code_tab_chart"]]
                        )
                        with _tab_a:
                            for _snip in _analysis:
                                st.code(_snip, language="python")
                        with _tab_c:
                            for _snip in _charts_code:
                                st.code(_snip, language="python")
                    elif _analysis:
                        for _snip in _analysis:
                            st.code(_snip, language="python")
                    else:
                        for _snip in _charts_code:
                            st.code(_snip, language="python")

        # ── Action icons + Quick-question chips ──────────────────────────
        _show_bottom = (
            (msg.get("is_system") and last_asst_idx == -1)
            or msg_idx == last_asst_idx
        )
        if _show_bottom:
            _render_action_icons(T)
            _render_quick_chips(T, context_aware=True)


# ─── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    init_state()
    lang = st.session_state.lang
    T = TEXT[lang]

    # Restore tools._df from session-state cache after module reload
    # (tools._df is module-level and resets to None on Streamlit file-change reruns)
    if st.session_state.get("data_loaded") and tools.get_dataframe() is None:
        _cached_df = st.session_state.get("_summary_df")
        if _cached_df is not None:
            _cached_name = st.session_state.get("_summary_name") or "dataset"
            tools.set_dataframe(_cached_df, _cached_name)
        else:
            # Truly lost — reset so the UI is consistent
            st.session_state.data_loaded = False

    # Restore AI dashboard charts from session state after module reload
    if st.session_state.get("ai_dashboard_charts") and not tools.get_ai_dashboard_charts():
        tools._ai_dashboard_charts = list(st.session_state.ai_dashboard_charts)

    # Regenerate the dataset-summary system message if language changed since it was created
    if (
        st.session_state.get("data_loaded")
        and st.session_state.get("_summary_lang") != lang
        and st.session_state.messages
        and st.session_state.messages[0].get("is_system")
    ):
        # Use session-state-cached df (survives module reloads; tools._df is module-level)
        _df = st.session_state.get("_summary_df")
        if _df is None:
            _df = tools.get_dataframe()
        if _df is not None:
            _ds_name = st.session_state.get("_summary_name") or tools.get_data_name() or "dataset"
            st.session_state.messages[0]["content"] = _make_dataset_summary(_df, _ds_name, T)
            st.session_state._summary_lang = lang

    inject_css(lang, theme=st.session_state.theme)
    # Skip-to-main link for keyboard / screen-reader users
    st.markdown(
        f'''<a class="skip-link" href="#main-content"
               onclick="(function(){{
                 var t=document.getElementById('main-content');
                 if(t){{t.focus();t.scrollIntoView({{behavior:'smooth',block:'start'}});}};
               }})(); return false;"
            >{T["a11y_skip"]}</a>''',
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
    st.markdown(f"""
<div class="app-header">
  <div class="app-title">{T["page_title"]}</div>
  <div class="app-sub">{T["page_sub"]}</div>
</div>""", unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────
    render_sidebar(T)

    # ── Ensure chat object exists ─────────────────────────────────────────
    if st.session_state.chat is None and st.session_state.data_loaded:
        try:
            st.session_state.chat = build_chat()
        except ValueError:
            pass  # Ollama not configured yet — chat stays None

    # ── Skip-link target anchor ───────────────────────────────────────────
    st.markdown(
        '<div id="main-content" tabindex="-1" '
        'style="outline:none;position:absolute;top:0;left:0;"></div>',
        unsafe_allow_html=True,
    )

    # ── 2 Tabs ────────────────────────────────────────────────────────────
    tab_chat, tab_ai_dash, tab_dashboard, tab_data = st.tabs([
        T["tab_chat"], T["tab_ai_dashboard"], T["tab_dashboard"], T["tab_data"]
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # Tab 1 – AI Chat
    # ═══════════════════════════════════════════════════════════════════════
    with tab_chat:
        _onboard_step = _get_onboarding_step()
        if _onboard_step != "done":
            # ── Onboarding: guide user through setup via chat messages ────
            _render_onboarding_chat(T)
        else:
            # ── Normal chat: data loaded + AI ready ───────────────────────

            # Dataset info badge (compact, inline)
            _df_badge = tools.get_dataframe()
            if _df_badge is not None:
                _ds_name = tools.get_data_name() or "dataset"
                st.caption(
                    T["dataset_badge_tpl"].format(
                        name=_ds_name,
                        rows=len(_df_badge),
                        cols=len(_df_badge.columns),
                    )
                )

            # Chat-just-loaded banner
            if st.session_state.get("chat_just_loaded"):
                st.info(T["chat_loaded_note"])
                st.session_state.chat_just_loaded = False

            # Welcome message + chips for empty conversation
            if not st.session_state.messages:
                with st.chat_message("assistant"):
                    st.markdown(T["chat_welcome"])
                _render_quick_chips(T, context_aware=False)
            else:
                render_history()

        # ── Recover from interrupted streaming ─────────────────────────
        if st.session_state._is_streaming and st.session_state._streaming_target == "chat":
            st.session_state._is_streaming = False
            st.session_state._streaming_target = None
            _partial = st.session_state._partial_response
            if _partial:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": _partial,
                    "charts": [],
                    "chart_configs": [],
                    "code_snippets": [],
                })
                st.session_state._partial_response = ""

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
        if no_key:
            st.caption(T["no_key_hint"])
        elif no_data:
            st.caption(T["no_data_warn"])

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
                # ── Thinking indicator: animated dots until first token ────
                _think_ph = st.empty()
                _think_ph.markdown(
                    "<div class='thinking-dots'>"
                    "<span></span><span></span><span></span>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                # Auto-scroll to show thinking dots
                import streamlit.components.v1 as _components
                _components.html("""<script>
                    const m = window.parent.document.querySelector('[data-testid="stMainBlockContainer"]')
                           || window.parent.document.querySelector('.main .block-container');
                    if (m) m.scrollTop = m.scrollHeight;
                </script>""", height=0)

                _cleared = [False]

                def _stream_with_clear():
                    """Wrap the token stream: clears the placeholder on 1st chunk."""
                    for _chunk in text_stream(st.session_state.chat, final_input):
                        if not _cleared[0]:
                            _think_ph.empty()
                            _cleared[0] = True
                        yield _chunk

                # ── Streaming: manual loop with stop support ─────────
                st.session_state._is_streaming = True
                st.session_state._partial_response = ""
                st.session_state._streaming_target = "chat"

                # CSS: morph send button → stop button (red ■)
                st.markdown("""<style>
                .stChatInput button svg { display: none !important; }
                .stChatInput button::before {
                    content: "\\25A0"; font-size: 16px; line-height: 1; color: white;
                }
                .stChatInput button { background: #e74c3c !important; }
                .stChatInput button:hover { background: #c0392b !important; }
                </style>""", unsafe_allow_html=True)

                # JS: auto-scroll + stop-button click handler
                _components.html("""<script>
                (function() {
                    const parent = window.parent.document;
                    function scrollToBottom() {
                        const main = parent.querySelector('[data-testid="stMainBlockContainer"]')
                                  || parent.querySelector('.main .block-container');
                        if (main) main.scrollTop = main.scrollHeight;
                    }
                    scrollToBottom();
                    const si = setInterval(scrollToBottom, 300);
                    const btn = parent.querySelector('.stChatInput button');
                    if (btn) {
                        btn.addEventListener('click', function(e) {
                            e.preventDefault(); e.stopPropagation();
                            const sb = parent.querySelector('[data-testid="stStatusWidget"] button');
                            if (sb) sb.click();
                        });
                    }
                    setTimeout(() => clearInterval(si), 120000);
                })();
                </script>""", height=0)

                _content_ph = st.empty()
                try:
                    full_text = _manual_stream(_stream_with_clear(), _content_ph)
                except Exception as err:
                    if not _cleared[0]:
                        _think_ph.empty()
                    full_text = st.session_state._partial_response or ""
                    if str(err) not in ("", "None"):
                        st.error(_friendly_error(err))
                    _audit("CHAT_ERROR", str(err))
                finally:
                    st.session_state._is_streaming = False
                    st.session_state._streaming_target = None
                _elapsed = (datetime.datetime.now() - _t0).total_seconds()
                chart_paths   = tools.get_pending_charts()
                chart_configs = tools.get_pending_chart_configs()
                code_snippets = tools.get_pending_code_snippets()
                for chart_path in chart_paths:
                    if chart_path.exists():
                        st.image(str(chart_path), use_container_width=True)

            _audit(
                "CHAT_RESPONSE",
                f"q_len={len(final_input)} resp_len={len(full_text or '')} "
                f"charts={len(chart_paths)} elapsed={_elapsed:.1f}s",
            )
            st.session_state.messages.append({
                "role":          "assistant",
                "content":       full_text or "",
                "charts":        [str(p) for p in chart_paths],
                "chart_configs": chart_configs,
                "code_snippets": code_snippets,
            })
            # Auto-save conversation to disk
            _msgs = st.session_state.messages
            _user_msgs = [m for m in _msgs if m["role"] == "user"]
            if _user_msgs:
                if not st.session_state.current_chat_id:
                    st.session_state.current_chat_id = _new_chat_id()
                _chat_name = _user_msgs[0]["content"][:60]
                try:
                    _save_chat(st.session_state.current_chat_id, _msgs, _chat_name)
                except Exception as _save_err:
                    _logger.warning("Chat save failed: %s", _save_err)
            # Chips: re-show after each AI response
            st.session_state.chips_hidden = False
            # Rerun so render_history() re-renders with the new message
            # (icons + chips appear after the last assistant reply)
            st.rerun()

        elif final_input and not st.session_state.chat:
            st.warning(T["no_data_warn"])

    # ═══════════════════════════════════════════════════════════════════════
    # Tab 2 – AI Dashboard  (chat-driven dashboard builder)
    # ═══════════════════════════════════════════════════════════════════════
    with tab_ai_dash:
        df_ai = tools.get_dataframe()
        if df_ai is None:
            st.info(T["ai_dash_no_data"])
        else:
            # ── Sync tools ↔ session state ────────────────────────────────
            if tools.get_ai_dashboard_charts():
                st.session_state.ai_dashboard_charts = tools.get_ai_dashboard_charts()
            elif st.session_state.ai_dashboard_charts:
                tools._ai_dashboard_charts = list(st.session_state.ai_dashboard_charts)

            col_chat_ai, col_preview = st.columns([2, 3])

            # ── LEFT COLUMN: Chat ─────────────────────────────────────────
            with col_chat_ai:
                # ── Recover from interrupted streaming ─────────────────
                if st.session_state._is_streaming and st.session_state._streaming_target == "ai_dash":
                    st.session_state._is_streaming = False
                    st.session_state._streaming_target = None
                    _partial = st.session_state._partial_response
                    if _partial:
                        st.session_state.ai_dash_messages.append({
                            "role": "assistant", "content": _partial,
                        })
                        st.session_state._partial_response = ""
                        st.session_state.ai_dashboard_charts = tools.get_ai_dashboard_charts()

                # Render chat history
                for _msg in st.session_state.ai_dash_messages:
                    with st.chat_message(_msg["role"]):
                        st.markdown(_msg["content"])

                # Welcome message
                if not st.session_state.ai_dash_messages:
                    with st.chat_message("assistant"):
                        st.markdown(T["ai_dash_welcome"])

                # Chat input
                if st.session_state.get("provider", "anthropic") == "ollama":
                    _ai_no_key = not (st.session_state.ollama_available
                                      and bool(st.session_state.ollama_model))
                else:
                    _ai_no_key = not bool(get_active_key())

                _ai_dash_input = st.chat_input(
                    T["ai_dash_input_ph"],
                    disabled=_ai_no_key,
                    key="ai_dash_chat_input",
                )

                if _ai_dash_input:
                    # Ensure chat object exists
                    if st.session_state.ai_dash_chat is None:
                        st.session_state.ai_dash_chat = build_ai_dash_chat()

                    # Display user message
                    with st.chat_message("user"):
                        st.markdown(_ai_dash_input)
                    st.session_state.ai_dash_messages.append({
                        "role": "user", "content": _ai_dash_input,
                    })

                    # Stream AI response
                    with st.chat_message("assistant"):
                        _think_ph2 = st.empty()
                        _think_ph2.markdown(
                            "<div class='thinking-dots'>"
                            "<span></span><span></span><span></span></div>",
                            unsafe_allow_html=True,
                        )
                        # Auto-scroll to thinking dots
                        import streamlit.components.v1 as _components
                        _components.html("""<script>
                            const m = window.parent.document.querySelector('[data-testid="stMainBlockContainer"]')
                                   || window.parent.document.querySelector('.main .block-container');
                            if (m) m.scrollTop = m.scrollHeight;
                        </script>""", height=0)

                        _cleared2 = [False]

                        def _ai_dash_stream():
                            for _ch in text_stream(st.session_state.ai_dash_chat, _ai_dash_input):
                                if not _cleared2[0]:
                                    _think_ph2.empty()
                                    _cleared2[0] = True
                                yield _ch

                        # ── Streaming: manual loop with stop support ─────
                        st.session_state._is_streaming = True
                        st.session_state._partial_response = ""
                        st.session_state._streaming_target = "ai_dash"

                        # CSS: morph send button → stop button (red ■)
                        st.markdown("""<style>
                        .stChatInput button svg { display: none !important; }
                        .stChatInput button::before {
                            content: "\\25A0"; font-size: 16px; line-height: 1; color: white;
                        }
                        .stChatInput button { background: #e74c3c !important; }
                        .stChatInput button:hover { background: #c0392b !important; }
                        </style>""", unsafe_allow_html=True)

                        # JS: auto-scroll + stop-button click handler
                        _components.html("""<script>
                        (function() {
                            const parent = window.parent.document;
                            function scrollToBottom() {
                                const main = parent.querySelector('[data-testid="stMainBlockContainer"]')
                                          || parent.querySelector('.main .block-container');
                                if (main) main.scrollTop = main.scrollHeight;
                            }
                            scrollToBottom();
                            const si = setInterval(scrollToBottom, 300);
                            const btn = parent.querySelector('.stChatInput button');
                            if (btn) {
                                btn.addEventListener('click', function(e) {
                                    e.preventDefault(); e.stopPropagation();
                                    const sb = parent.querySelector('[data-testid="stStatusWidget"] button');
                                    if (sb) sb.click();
                                });
                            }
                            setTimeout(() => clearInterval(si), 120000);
                        })();
                        </script>""", height=0)

                        _content_ph2 = st.empty()
                        try:
                            _ai_dash_text = _manual_stream(_ai_dash_stream(), _content_ph2)
                        except Exception as _err:
                            if not _cleared2[0]:
                                _think_ph2.empty()
                            _ai_dash_text = st.session_state._partial_response or ""
                            if str(_err) not in ("", "None"):
                                st.error(_friendly_error(_err))
                        finally:
                            st.session_state._is_streaming = False
                            st.session_state._streaming_target = None

                    st.session_state.ai_dash_messages.append({
                        "role": "assistant", "content": _ai_dash_text or "",
                    })

                    # Sync chart state from tools module into session state
                    st.session_state.ai_dashboard_charts = tools.get_ai_dashboard_charts()
                    st.rerun()

                # Control buttons
                _btn1, _btn2 = st.columns(2)
                with _btn1:
                    if st.button(T["ai_dash_clear_chat"], key="ai_dash_clr_chat",
                                 use_container_width=True):
                        st.session_state.ai_dash_messages = []
                        st.session_state.ai_dash_chat = None
                        st.rerun()
                with _btn2:
                    if st.button(T["ai_dash_clear"], key="ai_dash_clr_board",
                                 use_container_width=True):
                        st.session_state.ai_dashboard_charts = []
                        tools.clear_ai_dashboard_charts()
                        st.rerun()

            # ── RIGHT COLUMN: Dashboard Preview ───────────────────────────
            with col_preview:
                _ai_charts = st.session_state.ai_dashboard_charts
                if not _ai_charts:
                    st.markdown(
                        f"<div style='text-align:center;padding:80px 20px;opacity:0.5'>"
                        f"<div style='font-size:3rem'>🤖</div>"
                        f"<div style='font-size:1.2rem;margin-top:8px'>{T['ai_dash_empty_title']}</div>"
                        f"<div style='font-size:0.9rem;margin-top:4px'>{T['ai_dash_empty_hint']}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    # Header row
                    _h1, _h2, _h3, _h4 = st.columns([3, 2, 1, 1])
                    with _h1:
                        st.markdown(f"**{len(_ai_charts)} {T['ai_dash_charts_count']}**")
                    with _h2:
                        _ai_grid = st.radio(
                            T["ai_dash_layout"], [1, 2, 3],
                            horizontal=True, index=1,
                            format_func=lambda x: f"{x} {T['dash_col_suffix']}",
                            key="ai_dash_grid",
                        )
                    with _h3:
                        _html_export = export_dashboard_html(
                            _ai_charts, df_ai, T["tab_ai_dashboard"]
                        )
                        st.download_button(
                            T["ai_dash_export"],
                            data=_html_export.encode("utf-8"),
                            file_name="ai_dashboard.html",
                            mime="text/html",
                            use_container_width=True,
                            key="ai_dash_export_btn",
                        )
                    with _h4:
                        if st.button(T["ai_dash_copy_to_manual"], key="ai_dash_copy",
                                     use_container_width=True):
                            st.session_state.dashboard_charts.extend(_ai_charts)
                            st.toast(T["ai_dash_copied"])

                    st.markdown("---")

                    # Chart grid
                    _i = 0
                    while _i < len(_ai_charts):
                        _cols = st.columns(_ai_grid)
                        for _j in range(_ai_grid):
                            _idx = _i + _j
                            if _idx < len(_ai_charts):
                                _c = _ai_charts[_idx]
                                with _cols[_j]:
                                    _lbl = _c.get("title") or f"{_c.get('type', 'Chart')} – {_c.get('x', '')}"
                                    st.markdown(f"**{_lbl}**")
                                    try:
                                        _fig = build_chart(_c, df_ai, sample_size=_c.get("sample_size", 500))
                                        st.plotly_chart(_fig, use_container_width=True,
                                                        key=f"ai_dash_chart_{_idx}")
                                    except Exception as _e:
                                        st.error(f"Error: {_e}")
                                    if st.button(
                                        T["ai_dash_remove_chart"],
                                        key=f"ai_rm_{_idx}",
                                        use_container_width=True,
                                    ):
                                        st.session_state.ai_dashboard_charts.pop(_idx)
                                        tools._ai_dashboard_charts = list(
                                            st.session_state.ai_dashboard_charts
                                        )
                                        st.rerun()
                        _i += _ai_grid

    # ═══════════════════════════════════════════════════════════════════════
    # Tab 3 – Dashboard  (chart builder + charts grid + data view)
    # ═══════════════════════════════════════════════════════════════════════
    with tab_dashboard:
        df_now = tools.get_dataframe()
        if df_now is None:
            st.info(T["no_data_warn"])
        else:
            # ── AI Chart Style (collapsible) ──────────────────────────────
            _STYLES   = ["whitegrid", "darkgrid", "white", "ticks"]
            _PALETTES = ["husl", "deep", "muted", "Set2", "tab10", "viridis", "rocket"]
            _SIZES    = {"10×5": (10, 5), "12×6": (12, 6), "8×4": (8, 4), "14×7": (14, 7)}
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
                st.session_state.chart_ai_palette    = pal_val
                st.session_state.chart_figsize_lbl   = size_lbl
                tools.set_chart_style(
                    seaborn_style=style_val,
                    palette=pal_val,
                    figsize=_SIZES[size_lbl],
                )

            # ── Chart Builder (collapsible) ────────────────────────────────
            with st.expander(T["chart_builder_hdr"], expanded=False):
                num_c = df_now.select_dtypes(include="number").columns.tolist()
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    chart_type_sel = st.selectbox(T["chart_type"], _chart_types_for_lang(), key="cb_type")
                with col_b:
                    x_col_sel = st.selectbox(T["chart_x"], df_now.columns.tolist(), key="cb_x")
                with col_c:
                    y_col_sel = st.selectbox(T["chart_y"], ["—"] + num_c, key="cb_y")
                col_d, col_e, col_f = st.columns(3)
                with col_d:
                    color_sel = st.selectbox(T["chart_color_by"], ["—"] + df_now.columns.tolist(), key="cb_color")
                    chart_color = None if color_sel == "—" else color_sel
                with col_e:
                    palette_sel = st.selectbox(T["chart_palette"], list(_palettes_for_lang().keys()), key="cb_palette")
                with col_f:
                    chart_title_inp = st.text_input(
                        T["chart_title_lbl"], placeholder=T["chart_title_ph"], key="cb_title"
                    )
                corr_method_val = "pearson"
                if _norm_chart_type(chart_type_sel).split(" / ")[-1] == "Heatmap":
                    _corr_labels = (
                        {"pearson": "Pearson", "spearman": "Spearman", "kendall": "Kendall"}
                        if st.session_state.get("lang") == "en" else
                        {"pearson": "פירסון (Pearson)", "spearman": "ספירמן (Spearman)", "kendall": "קנדל (Kendall)"}
                    )
                    corr_method_val = st.selectbox(
                        T["chart_corr_method"],
                        ["pearson", "spearman", "kendall"],
                        format_func=lambda x: _corr_labels[x],
                        key="cb_corr",
                    )
                max_sample   = max(50, min(5_000, len(df_now)))
                chart_sample = st.slider(
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
                    st.error(f'{T["chart_builder_err"]}: {e}')
                _no_sample = {"Heatmap", "Box Plot", "Pie", "Histogram"}
                _sel_eng = _norm_chart_type(chart_type_sel).split(" / ")[-1]
                if _sel_eng not in _no_sample:
                    st.caption(T["chart_sample_note"].format(n=chart_sample, total=len(df_now)))
                # ── Python code for this chart ──────────────────────────────
                with st.expander(T["chart_code_label"], expanded=False):
                    st.code(_build_chart_code(cfg, chart_sample), language="python")
                if st.button(T["add_to_dash"], key="add_to_dashboard"):
                    cfg_to_save = {**cfg, "sample_size": chart_sample}
                    st.session_state.dashboard_charts.append(cfg_to_save)
                    n = len(st.session_state.dashboard_charts)
                    st.success(f'{T["chart_added"]} {T["chart_added_total"].format(n=n)}')

            st.markdown("---")

            # ── Dashboard grid ─────────────────────────────────────────────
            if not st.session_state.dashboard_charts:
                st.markdown(
                    f"""
<div class="dash-empty-state">
  <div class="dash-empty-icon">📊</div>
  <div class="dash-empty-title">{T["dash_empty_title"]}</div>
  <div class="dash-empty-steps">
    <div class="dash-step"><span class="step-num">1</span><span>{T["dash_empty_step1"]}</span></div>
    <div class="dash-step"><span class="step-num">2</span><span>{T["dash_empty_step2"]}</span></div>
    <div class="dash-step"><span class="step-num">3</span><span>{T["dash_empty_step3"]}</span></div>
  </div>
  <div class="dash-empty-tip">{T["dash_empty_tip"]}</div>
</div>
""",
                    unsafe_allow_html=True,
                )
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
                                if c.get("type") == "image":
                                    img_p = Path(c.get("path", ""))
                                    if img_p.exists():
                                        st.image(str(img_p), use_container_width=True)
                                    else:
                                        st.warning("תמונה לא נמצאה")
                                else:
                                    try:
                                        fig = build_chart(c, df_now, sample_size=c.get("sample_size", 500))
                                        st.plotly_chart(fig, use_container_width=True, key=f"dash_{idx}")
                                    except Exception as e:
                                        st.error(f"שגיאה: {e}")
                                if st.button(T["dash_remove"], key=f"rm_dash_{idx}", use_container_width=True):
                                    st.session_state.dashboard_charts.pop(idx)
                                    st.rerun()
                    i += grid_cols


    # ═══════════════════════════════════════════════════════════════════════
    # Tab 3 – Data View
    # ═══════════════════════════════════════════════════════════════════════
    with tab_data:
        df_data = tools.get_dataframe()
        if df_data is None:
            st.info(T["no_data_warn"])
        else:
            if st.session_state.data_warnings:
                with st.expander(
                    f"⚠️ {len(st.session_state.data_warnings)} {T['data_warnings_hdr']}",
                    expanded=True,
                ):
                    for w in st.session_state.data_warnings:
                        st.warning(w)
                    if st.button(T["data_autofix_btn"], key="auto_fix_data"):
                        fixed = auto_fix_dataframe(df_data)
                        tools.set_dataframe(fixed, name=tools.get_data_name())
                        st.session_state.data_warnings = []
                        st.success(T["data_fixed"])
                        st.rerun()
            col1, col2 = st.columns(2)
            with col1:
                search = st.text_input(
                    T["data_search"], placeholder=T["data_search_ph"], key="data_search_input"
                )
            with col2:
                filter_col = st.selectbox(
                    T["data_filter_col"],
                    [T["data_filter_all"]] + df_data.columns.tolist(),
                    key="data_filter_col_sel",
                )
            display_df = df_data.copy()
            if search:
                if filter_col != T["data_filter_all"] and filter_col in display_df.columns:
                    mask = display_df[filter_col].astype(str).str.contains(search, case=False, na=False)
                else:
                    mask = display_df.astype(str).apply(
                        lambda col: col.str.contains(search, case=False, na=False)
                    ).any(axis=1)
                display_df = display_df[mask]
            st.dataframe(display_df, use_container_width=True, height=400)
            cap_col, dl_col = st.columns([4, 1])
            with cap_col:
                st.caption(T["data_showing"].format(n=len(display_df), total=len(df_data)))
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
            if st.checkbox(T["data_stats_chk"], key="data_stats_cb"):
                num_only = df_data.select_dtypes(include="number")
                cat_only = df_data.select_dtypes(exclude="number")
                _tab_num, _tab_cat = st.tabs([T["stats_tab_numeric"], T["stats_tab_categ"]])
                with _tab_num:
                    if not num_only.empty:
                        st.dataframe(num_only.describe().round(2), use_container_width=True)
                    else:
                        st.info(T["no_numeric_cols"])
                with _tab_cat:
                    _cat_cols = [c for c in cat_only.columns if cat_only[c].nunique() > 1]
                    if _cat_cols:
                        for col in _cat_cols:
                            with st.expander(col, expanded=False):
                                # Raw value counts (all values)
                                vc = cat_only[col].value_counts()
                                if len(vc) > 30:
                                    vc = vc.head(30)
                                _raw_counts = {str(v): int(c) for v, c in vc.items()}
                                _values = list(_raw_counts.keys())

                                # Build editable DataFrame
                                freq_df = pd.DataFrame({
                                    "Exclude": [False] * len(_values),
                                    "Value":   _values,
                                    "Order":   list(range(1, len(_values) + 1)),
                                    "Count":   list(_raw_counts.values()),
                                    "%":       [0.0] * len(_values),
                                })
                                # Calculate initial %
                                _total_init = sum(_raw_counts.values())
                                if _total_init > 0:
                                    freq_df["%"] = (freq_df["Count"] / _total_init * 100).round(1)

                                # Editable table
                                edited = st.data_editor(
                                    freq_df,
                                    key=f"cat_editor_{col}",
                                    use_container_width=True,
                                    hide_index=True,
                                    disabled=["Value", "Count", "%"],
                                    column_config={
                                        "Exclude": st.column_config.CheckboxColumn(
                                            "Exclude", default=False,
                                        ),
                                        "Order": st.column_config.NumberColumn(
                                            "Order", min_value=1,
                                        ),
                                    },
                                )

                                # Recalculate: excluded rows get "—", rest get recalculated %
                                _included_mask = ~edited["Exclude"]
                                _included_total = edited.loc[_included_mask, "Count"].sum()
                                _result = edited.copy()
                                if _included_total > 0:
                                    _result.loc[_included_mask, "%"] = (
                                        _result.loc[_included_mask, "Count"] / _included_total * 100
                                    ).round(1)
                                _result.loc[~_included_mask, "Count"] = None
                                _result.loc[~_included_mask, "%"] = None

                                # Sort by Order and display
                                _result = _result.sort_values("Order").reset_index(drop=True)
                                st.dataframe(
                                    _result[["Exclude", "Value", "Order", "Count", "%"]],
                                    use_container_width=True,
                                    hide_index=True,
                                )
                                st.caption(T["cat_order_hint"])
                    else:
                        st.info(T["no_categ_cols"])

if __name__ == "__main__":
    main()
