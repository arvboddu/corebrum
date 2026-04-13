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
echo Copilot launch requested.
echo.
echo Audio routing reminder:
echo 1. Open Windows Settings ^> System ^> Sound ^> Volume mixer.
echo 2. Set your Zoom or Teams meeting output device to CABLE Input ^(VB-Audio Virtual Cable^).
echo 3. Keep your regular speakers or headphones as your main listening device if needed.
echo 4. The backend listens on CABLE Output ^(VB-Audio Virtual Cable^) and forwards answers to the HUD.
echo.
echo If the HUD does not respond, check:
echo - runtime-logs\backend.out.log
echo - runtime-logs\backend.err.log
echo - that Ollama is open and llama3.1 is installed

endlocal
