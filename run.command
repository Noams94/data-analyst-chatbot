#!/bin/bash
# ─────────────────────────────────────────────
#  Data Analyst Chatbot — Launcher
#  לחץ פעמיים על הקובץ הזה כדי להפעיל את האפליקציה
# ─────────────────────────────────────────────

# Move to the folder where this script lives
cd "$(dirname "$0")"

echo ""
echo "======================================"
echo " 🤖 Data Analyst Chatbot"
echo "======================================"
echo ""

# ── Check Python ──────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "❌  Python3 not found. Install from https://python.org"
  read -p "Press Enter to close..."
  exit 1
fi

# ── Install dependencies if needed ────────
echo "🔍  Checking dependencies..."
if ! python3 -c "import streamlit" &>/dev/null; then
  echo "📦  Installing dependencies (first run only)..."
  pip3 install -r requirements.txt
  echo ""
fi

# ── Check API key ─────────────────────────
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "⚠️  ANTHROPIC_API_KEY is not set."
  echo "    Enter your API key (or press Enter to skip if already configured elsewhere):"
  read -r key
  if [ -n "$key" ]; then
    export ANTHROPIC_API_KEY="$key"
  fi
fi

# ── Launch ────────────────────────────────
echo ""
echo "🚀  Starting the app..."
echo "    Opening http://localhost:8501 in your browser"
echo "    Press Ctrl+C to stop"
echo ""

python3 -m streamlit run app.py \
  --server.port 8501 \
  --browser.gatherUsageStats false
