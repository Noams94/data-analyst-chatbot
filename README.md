# Data Analyst Chatbot

A bilingual (Hebrew / English) AI-powered data analysis tool built with **Streamlit** and multi-provider LLM support.

Upload a CSV, Excel, JSON, or Parquet file and ask questions in natural language — the bot generates charts, summaries, and interactive dashboards.

---

## Features

| Feature | Description |
|---|---|
| AI Chat | Ask questions in Hebrew or English; the bot runs pandas analyses and creates charts automatically |
| AI Dashboard | Chat-driven dashboard builder — describe charts in natural language and the AI creates, modifies, and arranges them |
| Chart Builder | Interactive Plotly chart builder with 8 chart types and 16 colour palettes |
| Dashboard | Collect charts into a customisable multi-column dashboard |
| Data View | Browse, search, filter, and auto-fix data quality issues |
| Streaming Control | Live streaming with a stop button — cancel AI responses mid-stream |
| Export | Download conversation as HTML or PDF, AI charts as ZIP, dashboard as interactive HTML |
| Email | Send the conversation or data CSV via Gmail (App Password) |
| Dark Mode | Dark theme by default (light mode is built-in but disabled — see [Enabling Light Mode](#enabling-light-mode)) |
| Multi-Provider | Anthropic Claude, OpenAI, Google Gemini, Groq, or local Ollama |

---

## Supported LLM Providers

| Provider | Models | API Key Required |
|---|---|---|
| Anthropic | Claude Opus, Sonnet, Haiku | Yes |
| OpenAI | GPT-4o, GPT-4, GPT-3.5 | Yes |
| Google | Gemini Pro, Flash | Yes |
| Groq | Llama, Mixtral | Yes |
| Ollama | Any local model | No (runs locally) |

---

## Quick Start

### 1 - Local (Python 3.9+)

```bash
git clone https://github.com/Noams94/data-analyst-chatbot.git
cd data-analyst-chatbot
pip install -r requirements.txt
streamlit run app.py
```

Set your API key in the sidebar or export it as an environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

### 2 - Docker

```bash
docker compose up --build
# -> http://localhost:8501
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
| `.streamlit/secrets.toml` | Store API keys securely (not committed to git) |

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

## Tabs

### Chat
The main conversational interface. Upload a file, ask questions, and the AI runs analyses, generates charts, and provides insights. Includes a stop button to cancel streaming responses mid-generation.

### AI Dashboard
A chat-driven dashboard builder. Describe what you want in natural language (e.g., "create a bar chart of sales by region") and the AI uses tool-calling to create, modify, reorder, and delete charts. Charts are rendered as interactive Plotly visualisations with drag-and-drop arrangement.

### Dashboard
Manual chart builder with 8 chart types (Bar, Line, Area, Pie, Histogram, Scatter, Box Plot, Heatmap) and 16 colour palettes. Add charts to a multi-column dashboard layout and export as interactive HTML.

### Data
Browse the uploaded dataset with search, filtering, and data quality diagnostics.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Tech Stack

- **Frontend:** Streamlit
- **AI:** chatlas + Anthropic / OpenAI / Google / Groq / Ollama
- **Data:** pandas, numpy
- **Charts:** matplotlib + seaborn (AI chat), Plotly (builder, dashboard, AI dashboard)
- **PDF:** fpdf2 + python-bidi (Hebrew RTL support)
- **Email:** smtplib (stdlib)

### chatlas — Multi-Provider AI Layer

The app uses [**chatlas**](https://github.com/posit-dev/chatlas) (by Posit) as a unified interface for all LLM providers. Instead of writing separate code per provider, chatlas provides one consistent API for chat, tool-calling, and streaming.

**How it works:**

1. **Create a Chat object** with the selected provider and system prompt:
   ```python
   from chatlas import ChatAnthropic, ChatOpenAI, ChatGoogle, ChatGroq, ChatOllama

   chat = ChatAnthropic(model="claude-sonnet-4-20250514", system_prompt=SYSTEM_PROMPT, api_key=key)
   ```

2. **Register tools** — plain Python functions become LLM-callable tools automatically (chatlas reads docstrings as descriptions and type hints as parameter schemas):
   ```python
   chat.register_tool(tools.get_data_overview)
   chat.register_tool(tools.run_analysis)
   chat.register_tool(tools.create_chart)
   ```

3. **Stream responses** — `chat.stream(prompt)` yields text chunks and tool calls; tools execute automatically as side-effects:
   ```python
   for chunk in chat.stream(prompt):
       if isinstance(chunk, str):
           yield chunk  # display to user
       # tool calls are executed automatically by chatlas
   ```

**Provider mapping** (in `build_chat()` and `build_ai_dash_chat()`):

| User selection | chatlas class | Notes |
|---|---|---|
| Anthropic Claude | `ChatAnthropic` | API key required |
| OpenAI | `ChatOpenAI` | API key required |
| Google Gemini | `ChatGoogle` | API key required |
| Groq | `ChatGroq` | API key required |
| Ollama (local) | `ChatOllama` | No key, runs locally |

Switching providers only changes which class is instantiated — all tool registration, streaming, and conversation management works identically.

---

## Analysis Architecture

### Tool Pipeline

The AI uses **4 registered tools** via function-calling (managed by `chatlas`):

| # | Tool | Purpose |
|---|------|---------|
| 1 | `get_data_overview()` | Initial scan — schema, dtypes, nulls, stats, sample rows |
| 2 | `run_analysis(pandas_code)` | Execute pandas / numpy code on the dataset |
| 3 | `create_chart(...)` | Generate matplotlib + seaborn visualisations (PNG) |
| 4 | `suggest_next_analyses()` | Propose smart follow-up questions |

#### AI Dashboard Tools

The AI Dashboard tab uses a separate chat object with **5 dashboard-specific tools**:

| # | Tool | Purpose |
|---|------|---------|
| 1 | `get_data_overview()` | Same initial scan as AI Chat |
| 2 | `set_dashboard_charts(charts)` | Replace the entire dashboard with a list of chart configs |
| 3 | `add_dashboard_chart(chart)` | Add a single chart to the dashboard |
| 4 | `update_dashboard_chart(index, updates)` | Modify a specific chart by 0-based index |
| 5 | `remove_dashboard_chart(index)` | Remove a chart by 0-based index |

### Chain-of-Thought Process

Every AI response follows a strict order enforced by the system prompt:

1. **Understand** — call `get_data_overview()` to inspect the dataset structure
2. **Compute** — call `run_analysis()` with pandas code to calculate exact numbers
3. **Report** — write the answer using only numbers produced by the analysis
4. **Visualise** — call `create_chart()` for every key finding

### Code Execution — `run_analysis()`

Uses a split exec / eval pattern in a sandboxed namespace:

- The DataFrame is **copied** before execution (original data is never mutated)
- **No access** to the filesystem, network, or OS commands
- Results are capped at **50 rows x 20 columns**
- All executed code is saved and displayed to the user in the Code panel

### Chart Generation — `create_chart()`

**Supported chart types:** bar, barh, line, scatter, hist, box, pie, heatmap, count

**Parameters:** `x_column`, `y_column`, `aggregation` (sum / mean / count / median / max / min), `color_column`, `top_n`, `bins`, `sort`

### Dashboard Charts (Plotly)

The interactive chart builder and dashboards use Plotly with modern styling:

- **16 colour palettes** — sequential, diverging, and qualitative
- **Custom hover tooltips** — per-chart-type templates with thousands separators
- **Animated entry** — CSS fade-in on chart mount + Plotly transitions
- **Theme-aware** — all colours adapt to dark / light mode

### Guardrails

The system prompt enforces strict rules:

- **Never** fabricate or estimate numbers — only report computed results
- **Never** modify, delete, or write back to the user's data
- **Never** access external URLs, files, APIs, or OS commands
- If unsure about a number, run additional code to verify

---

## System Prompts

Both tabs use separate system prompts that define the AI's role, process, response format, and guardrails. These are defined in `app.py` as `SYSTEM_PROMPT` and `AI_DASHBOARD_SYSTEM_PROMPT`.

### AI Chat — `SYSTEM_PROMPT`

```
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
1. Key Findings — bullet points with the exact numbers from run_analysis()
2. Interpretation — what the numbers mean in context
3. Insight / Recommendation — business or analytical conclusion
4. Follow-up — 1-2 natural follow-up questions

## Few-Shot Example
User: "What are the top 3 products by revenue?"
Thought process:
  -> call get_data_overview() to confirm column names
  -> call run_analysis("df.groupby('product')['revenue'].sum().nlargest(3)")
  -> call create_chart(chart_type='barh', x_column='product', y_column='revenue',
                       title='Top 3 Products by Revenue')
Response:
  Key Findings
  - Laptop: $45,230 (32% of total)
  - Phone: $38,100 (27%)
  - Tablet: $21,500 (15%)
  Interpretation — Electronics dominate revenue; Laptop alone accounts for nearly a third.
  Insight — Consider bundling Laptop with accessories to increase average basket size.
  Follow-up — Want to see revenue trend by month for these products?

## Chart Guidelines
- barh    -> many/long category names
- line    -> time series or ordered x-axis
- scatter -> two numeric variables
- heatmap -> correlations
- hist    -> distributions
- Always set a clear, descriptive title in the user's language

## Style
- Be concise yet complete
- Round numbers to 2 decimal places
- When writing Hebrew, keep markdown formatting clean
```

### AI Dashboard — `AI_DASHBOARD_SYSTEM_PROMPT`

```
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
```

---

## Security Notes

- API keys are stored in-memory only (session state) and never written to disk
- The app password for email is never persisted between sessions
- User input is capped at **2,000 characters** per message
- AI requests are rate-limited to **30 per session**
- Files larger than **100,000 rows** are automatically sampled

---

## Enabling Light Mode

The app ships in **dark-only mode**, but all light mode code is already built-in and ready to activate. To enable the light/dark theme toggle:

### Step 1 — Uncomment the toggle button in `app.py`

Search for `Theme toggle (disabled` (~line 3601) and uncomment the 4 lines:

```python
# Before (disabled):
# _theme_label = T["theme_btn_light"] if st.session_state.theme == "dark" else T["theme_btn_dark"]
# if st.button(_theme_label, use_container_width=True, key="theme_toggle"):
#     st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
#     st.rerun()

# After (enabled):
_theme_label = T["theme_btn_light"] if st.session_state.theme == "dark" else T["theme_btn_dark"]
if st.button(_theme_label, use_container_width=True, key="theme_toggle"):
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.rerun()
```

### Step 2 — Update `.streamlit/config.toml`

Change the theme base and colours to light:

```toml
[theme]
base                     = "light"
primaryColor             = "#10a37f"
backgroundColor          = "#ffffff"
secondaryBackgroundColor = "#f5f5f5"
textColor                = "#1a1a1a"
```

### Step 3 — Restart Streamlit

```bash
streamlit run app.py
```

A **☀️ Light Mode / 🌙 Dark Mode** toggle button will appear at the bottom of the sidebar. All colours, charts, and native components (including DataFrames) will adapt automatically.

---

## License

MIT
