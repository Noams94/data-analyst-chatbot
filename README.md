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

## Analysis Architecture

### Tool Pipeline

The AI uses **4 registered tools** via Claude's function-calling API (managed by `chatlas`):

| # | Tool | Purpose |
|---|------|---------|
| 1 | `get_data_overview()` | Initial scan — schema, dtypes, nulls, stats, sample rows |
| 2 | `run_analysis(pandas_code)` | Execute pandas / numpy code on the dataset |
| 3 | `create_chart(...)` | Generate matplotlib + seaborn visualisations (PNG) |
| 4 | `suggest_next_analyses()` | Propose smart follow-up questions |

### Chain-of-Thought Process

Every AI response follows a strict order enforced by the system prompt:

1. **Understand** — call `get_data_overview()` to inspect the dataset structure
2. **Compute** — call `run_analysis()` with pandas code to calculate exact numbers
3. **Report** — write the answer using **only** numbers produced by the analysis
4. **Visualise** — call `create_chart()` for every key finding

### Code Execution — `run_analysis()`

Uses a **split exec / eval** pattern in a sandboxed namespace:

```python
# Sandboxed namespace — only pandas, numpy, json + a copy of the data
local_ns = {"df": _df.copy(), "pd": pd, "np": np, "json": json}

# Execute all lines except the last
exec("\n".join(lines[:-1]), local_ns)

# Evaluate the last line to capture the result
result = eval(lines[-1], local_ns)
```

- The DataFrame is **copied** before execution (original data is never mutated)
- **No access** to the filesystem, network, or OS commands
- Results are capped at **50 rows × 20 columns**
- All executed code is saved and displayed to the user in the Code panel

### Chart Generation — `create_chart()`

**Supported chart types:** `bar` · `barh` · `line` · `scatter` · `hist` · `box` · `pie` · `heatmap` · `count`

**Parameters:** `x_column`, `y_column`, `aggregation` (sum / mean / count / median / max / min), `color_column`, `top_n`, `bins`, `sort`

- RTL (Hebrew) text rendered correctly via `python-bidi`
- Charts saved as PNG to the `charts/` directory
- Month-name axes are sorted chronologically

### Guardrails

The system prompt enforces strict rules:

- **Never** fabricate or estimate numbers — only report computed results
- **Never** modify, delete, or write back to the user's data
- **Never** access external URLs, files, APIs, or OS commands
- If unsure about a number → "I need to verify" + run additional code

### Response Format

Every AI response is structured as:

1. **📊 Key Findings** — bullet points with exact numbers
2. **🔍 Interpretation** — what the numbers mean in context
3. **💡 Insight / Recommendation** — business or analytical conclusion
4. **➡️ Follow-up** — 1–2 suggested next questions

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
