@echo off
rem RetailSense retailer app: one-click start on Windows.
rem Needs Python 3.11 and Node.js installed; for the critic, Ollama (or a key in .env, see configs\service.yaml).
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo First run: creating the Python environment...
  rem uv fetches Python 3.11 itself; otherwise a Python 3.11 known to the py launcher
  where uv >nul 2>nul && (
    uv venv -q --python 3.11 .venv && uv pip install -q --python .venv\Scripts\python.exe -e .
  ) || (
    py -3.11 -m venv .venv && .venv\Scripts\python -m pip install -q -e .
  ) || (
    if exist .venv rmdir /s /q .venv
    echo Setup failed. Install uv ^(docs.astral.sh/uv^) or Python 3.11 ^(python.org^), then run again.
    pause & exit /b 1
  )
)

if not exist web\out\index.html (
  echo First run: building the web app...
  where npm >nul 2>nul || (echo Node.js not found: install it from nodejs.org & pause & exit /b 1)
  pushd web
  call npm ci || (popd & echo npm ci failed & pause & exit /b 1)
  call npm run build || (popd & echo Building the web app failed & pause & exit /b 1)
  popd
)

curl -s -m 2 http://localhost:11434/api/tags >nul 2>nul || echo Note: Ollama is not running, so the critic falls back to the consensus forecast. Start Ollama, or set another LLM in configs\service.yaml.

.venv\Scripts\python -m retailsense.service.app
pause
