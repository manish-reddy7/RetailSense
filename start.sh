#!/usr/bin/env bash
# RetailSense retailer app: one-command start on macOS / Linux.
# Needs Python 3.11 and Node.js; for the critic, Ollama (or a key in .env, see configs/service.yaml).
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "First run: creating the Python environment..."
  python3.11 -m venv .venv
  .venv/bin/python -m pip install -q -e .
fi

if [ ! -f web/out/index.html ]; then
  echo "First run: building the web app..."
  (cd web && npm ci && npm run build)
fi

curl -s -m 2 http://localhost:11434/api/tags >/dev/null 2>&1 || echo "Note: Ollama is not running, so the critic falls back to the consensus forecast. Start Ollama, or set another LLM in configs/service.yaml."

exec .venv/bin/python -m retailsense.service.app
