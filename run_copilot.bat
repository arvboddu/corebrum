@echo off
setlocal

cd /d "%~dp0"

echo ==========================================
echo Corebrum Interview Copilot Launcher
echo ==========================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo Project virtual environment not found.
  echo Run setup_python311.bat first.
  exit /b 1
)

where ollama >nul 2>nul
if errorlevel 1 (
  echo Ollama is not installed or not on PATH.
  echo Install Ollama, then rerun this script.
  exit /b 1
)

echo Checking Ollama daemon...
ollama list >nul 2>nul
if errorlevel 1 (
  echo Ollama is not responding.
  echo Start the Ollama app, wait a few seconds, then rerun this script.
  exit /b 1
)

echo Checking Ollama model llama3.2:3b...
ollama show llama3.2:3b >nul 2>nul
if errorlevel 1 (
  echo Ollama model 'llama3.2:3b' is not installed.
  echo Run: ollama pull llama3.2:3b
  exit /b 1
)

if not exist "runtime-logs" mkdir "runtime-logs"

echo Stopping existing backend on port 8001 if present...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8001" ^| findstr "LISTENING"') do (
  taskkill /PID %%P /F >nul 2>nul
)

echo Starting Python backend in a hidden window...
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "Set-Location '%CD%'; & '.\.venv\Scripts\python.exe' -u 'engine\app.py' >> 'runtime-logs\backend.out.log' 2>> 'runtime-logs\backend.err.log'"

echo Waiting for backend startup...
timeout /t 6 /nobreak >nul

echo Starting Electron HUD...
start "" npm.cmd start

echo.
echo Copilot is starting...
echo.
echo Setup instructions:
echo 1. Open Chrome and load the extension from corebrum-extension\ folder:
echo    - Go to chrome://extensions/
echo    - Enable "Developer mode"
echo    - Click "Load unpacked" and select the corebrum-extension folder
echo 2. Click the extension icon and click "Start Capture"
echo    - Grant microphone permissions when prompted
echo 3. The HUD will display transcribed audio and AI suggestions
echo.
echo If the HUD does not respond, check:
echo - runtime-logs\backend.out.log
echo - runtime-logs\backend.err.log
echo - that Ollama is running with llama3.2:3b model installed

endlocal
