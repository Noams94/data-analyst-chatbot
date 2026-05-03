#!/bin/bash
source "$(dirname "$0")/.venv/bin/activate"
exec uvicorn api.main:app --reload --port 8001
