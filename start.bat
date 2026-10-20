@echo off
rem RetailSense retailer app: one-click start on Windows.
rem Needs Python 3.11 and Node.js installed; for the critic, Ollama (or a key in .env, see configs\service.yaml).
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo First run: creating the Python environment...
  py -3.11 -m venv .venv || (echo Python 3.11 not found: install it from python.org & pause & exit /b 1)
  .venv\Scripts\python -m pip install -q -e . || (echo Installing Python packages failed & pause & exit /b 1)
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
