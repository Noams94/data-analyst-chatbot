# 🤖 Data Analyst Chatbot

A bilingual (Hebrew / English) AI-powered data analysis tool built with **Streamlit** and **Claude Opus 4.6** (or local **Ollama** models).

Upload a CSV, Excel, JSON, or Parquet file and ask questions in natural language — the bot generates charts, summaries, and interactive dashboards.

---

## Features

| Feature | Description |
|---|---|
| 🤖 AI Chat | Ask questions in Hebrew or English; the bot runs pandas analyses and creates charts automatically |
| 📊 Chart Builder | Interactive Plotly chart builder with 8 chart types and colour palettes |
| 📈 Dashboard | Collect charts into a customisable multi-column dashboard |
| 🗂 Data View | Browse, search, filter, and auto-fix data quality issues |
| 📤 Export | Download conversation as **HTML** or **PDF**, AI charts as **ZIP**, dashboard as interactive **HTML** |
| 📧 Email | Send the conversation or data CSV via Gmail (App Password) |
| 🌙 Dark mode | Automatically respects your OS colour-scheme preference |
| 🦙 Ollama | Run fully offline with a local Ollama model (no API key needed) |

---

## Quick Start

### 1 · Local (Python 3.9+)

```bash
git clone https://github.com/Noams94/data-analyst-chatbot.git
cd data-analyst-chatbot
pip install -r requirements.txt
streamlit run app.py
```

Set your Anthropic API key in the sidebar **or** export it as an environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

### 2 · Docker

```bash
docker compose up --build
# → http://localhost:8501
```

Pass your API key via a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Configuration

| File | Purpose |
|---|---|
| `.streamlit/config.toml` | Theme colours, max upload size (200 MB), headless mode |
| `.streamlit/secrets.toml` | Store `ANTHROPIC_API_KEY` securely (not committed to git) |

**`secrets.toml` example:**
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

---

## Project Structure

```
data_analyst_chatbot/
├── app.py              # Main Streamlit application
├── tools.py            # Analysis tools: charts, pandas, overview
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .streamlit/
│   └── config.toml
└── tests/
    └── test_tools.py   # pytest unit tests
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Tech Stack

- **Frontend:** Streamlit
- **AI:** chatlas + Anthropic Claude Opus 4.6 (or Ollama)
- **Data:** pandas, numpy
- **Charts:** matplotlib / seaborn (AI), Plotly (builder & dashboard)
- **PDF:** fpdf2 + python-bidi (Hebrew RTL support)
- **Email:** smtplib (stdlib)

---

## Security Notes

- API keys are stored **in-memory only** (session state) and never written to disk
- The app password for email is never persisted between sessions
- User input is capped at **2,000 characters** per message
- AI requests are rate-limited to **30 per session**
- Files larger than **100,000 rows** are automatically sampled

---

## License

MIT
