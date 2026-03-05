# 🤖 Data Analyst Chatbot

צ'אטבוט ניתוח נתונים חכם — מופעל על ידי Claude AI ו-chatlas
Bilingual (Hebrew / English) data analysis chatbot powered by Claude AI & chatlas.

---

## ✨ פיצ'רים | Features

| לשונית | תיאור |
|---|---|
| 🤖 **AI Chat** | שאל שאלות בעברית או אנגלית — קבל תובנות, גרפים ו-follow-ups אוטומטיים |
| 📊 **גרפים** | בנה 8 סוגי גרפים אינטראקטיביים (Plotly) עם פלטות צבע ובקרת מדגם |
| 📈 **דשבורד** | הוסף גרפים לדשבורד אישי בפריסת 1/2/3 עמודות |
| 🗂 **נתונים** | חיפוש וסינון, אזהרות איכות נתונים + תיקון אוטומטי, סטטיסטיקות |

### יכולות AI
- 📊 ניתוח ותשובות מיידיות על הנתונים
- 📈 יצירת גרפים (matplotlib) ישירות מהשיחה
- 🔍 גילוי תובנות ומגמות
- 💡 הצעות לניתוחים המשך

---

## 🚀 הפעלה מהירה | Quick Start

### macOS — לחץ פעמיים
```bash
# תן הרשאת הרצה (פעם ראשונה בלבד)
chmod +x run.command
# ואז לחץ פעמיים על הקובץ run.command
```

### Terminal
```bash
pip install -r requirements.txt
streamlit run app.py
# פתח http://localhost:8501
```

---

## 🔑 מפתח API

1. קבל מפתח חינמי בכתובת: [console.anthropic.com](https://console.anthropic.com)
2. הזן אותו בסרגל הצדדי של האפליקציה (נשמר בזיכרון בלבד)
3. לחלופין, הגדר משתנה סביבה: `export ANTHROPIC_API_KEY=sk-ant-...`

---

## 📁 מבנה הפרויקט | Structure

```
data_analyst_chatbot/
├── app.py              ← Streamlit UI (bilingual, 4 tabs)
├── tools.py            ← Analysis tools: pandas, matplotlib charts
├── requirements.txt    ← Python dependencies
├── run.command         ← macOS double-click launcher
└── charts/             ← Generated chart images (git-ignored)
```

---

## 📦 תלויות | Dependencies

```
chatlas>=0.15.0     # LLM chat framework
anthropic>=0.49.0   # Anthropic SDK
streamlit>=1.40.0   # Web UI
pandas>=2.0.0       # Data processing
plotly>=5.0.0       # Interactive charts (dashboard)
matplotlib>=3.7.0   # AI-generated charts
seaborn>=0.12.0     # Chart styling
numpy>=1.24.0       # Numerical computing
openpyxl>=3.1.0     # Excel support
```

---

## 🔒 אבטחה ופרטיות | Security & Privacy

- מפתח ה-API נשמר **בזיכרון בלבד** — לא נכתב לדיסק
- הנתונים שלך **לא נשלחים לשרת** — רק סיכום סטטיסטי קצר נשלח ל-API לצורך שאלות AI
- הקוד בקונסולה רץ **מקומית בלבד** על המחשב שלך

---

## 🛠 טכנולוגיות | Tech Stack

- **[chatlas](https://posit-dev.github.io/chatlas/)** — Python LLM framework by Posit
- **[Claude Opus 4.6](https://anthropic.com)** — Anthropic AI model
- **[Streamlit](https://streamlit.io)** — Python web app framework
- **[Plotly](https://plotly.com)** — Interactive visualization

---

*Built with ❤️ using Claude AI*
