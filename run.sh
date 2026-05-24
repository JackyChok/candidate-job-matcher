#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Use venv if it exists, otherwise fall back to system python
if [ -f ".venv/bin/uvicorn" ]; then
  .venv/bin/uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
else
  uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
fi
