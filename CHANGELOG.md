# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- Team-wide hardening: caching, security, dark mode, Docker, tests, docs

---

## [0.6.0] — 2026-03-05

### Added — Performance (Developer)
- `@st.cache_data(ttl=30)` on `get_ollama_models()` — avoids repeated HTTP calls
- `@st.cache_data` on `validate_dataframe()` — re-uses results for unchanged DataFrames

### Added — Security (Security Engineer)
- `st.secrets` fallback for `ANTHROPIC_API_KEY` (supports Streamlit Cloud deployment)
- User input capped at **2,000 characters** per message (injection/cost protection)
- **Rate limiting**: max 30 AI requests per session; warning at 80 %, hard block at 100 %
- Large file auto-sampling: files > 100,000 rows are randomly sampled and a notice is shown

### Added — UX / Design
- **Dark mode** CSS via `@media (prefers-color-scheme: dark)` — custom elements adapt to OS theme
- **Onboarding hint** displayed after file upload (before first message) explaining the 4 tabs
- Streamlit theme configured in `.streamlit/config.toml` (brand blue, clean light palette)

### Added — DevOps
- `Dockerfile` with health-check (`/_stcore/health`)
- `docker-compose.yml` — single-command `docker compose up --build`
- `.dockerignore` — excludes tests, git, secrets, cache
- `.streamlit/config.toml` — max upload 200 MB, headless=true for containers

### Added — Tests
- `tests/test_tools.py` — 20 pytest unit tests covering:
  - `set_dataframe` / `get_dataframe` / `get_data_name`
  - `get_data_overview` content and type
  - `run_analysis` (valid code, invalid code, no data)
  - `_fix_rtl` (English, Hebrew, empty string)
  - `_sort_chronological` (English months, Hebrew months, non-month columns)
  - `create_chart` (bar, line, hist, pie, count, scatter, no data)
  - `suggest_next_analyses`

### Added — Docs
- `README.md` fully rewritten with feature table, Quick Start (local + Docker), project structure, security notes
- `CHANGELOG.md` (this file)

---

## [0.5.0] — 2026-03-05

### Added
- **PDF export** (`export_chat_pdf`) using fpdf2 + DejaVu Sans (Hebrew RTL via python-bidi)
- **Email send** (`send_email_smtp`) via Gmail SMTP + App Password (stdlib smtplib)
- Sidebar email expander: From / App Password / To / Subject / what-to-send radio
- `fpdf2>=2.7.0` added to `requirements.txt`

### Fixed
- `send_email_smtp` signature used `str | None` union syntax incompatible with Python 3.9 → replaced with bare defaults

---

## [0.4.0] — 2026-03-05

### Added
- **Responsive mobile + web UI/UX**: 3 CSS breakpoints (< 480 px / ≥ 768 px / ≥ 1024 px)
- Touch-friendly button min-height 44–48 px (Apple/Google HIG)
- Download buttons: distinct blue-tint accent
- Metric cards: `flex-wrap` so they stack on small screens
- Chat input: larger font + min-height for mobile keyboards
- `initial_sidebar_state="auto"` — sidebar auto-collapses on mobile

---

## [0.3.0] — 2026-03-05

### Added
- HTML conversation export with base64-embedded chart images
- Interactive dashboard HTML export (Plotly CDN, no kaleido needed)
- AI charts ZIP export (stdlib zipfile)
- Export row in Chat tab (HTML + PDF + ZIP buttons)
- Export button in Dashboard tab header

---

## [0.2.0] — 2026-03-04

### Added
- Hebrew RTL rendering in matplotlib charts via `python-bidi`
- Month chronological ordering (`_sort_chronological`)
- AI chart style UI in sidebar: seaborn style, palette, figure size

---

## [0.1.0] — 2026-03-04

### Added
- Initial release: 4-tab layout (AI Chat, Charts, Dashboard, Data)
- chatlas + Claude Opus 4.6 integration
- Ollama local model support
- Bilingual Hebrew / English UI
- CSV / Excel / JSON / Parquet file upload
- Interactive Plotly chart builder
- Data quality warnings and auto-fix
- CSV export
