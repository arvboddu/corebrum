@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found at .venv\Scripts\python.exe
  echo Run setup_python311.bat first.
  exit /b 1
)

echo Starting audio handler test from %CD%
".venv\Scripts\python.exe" -u "engine\audio_handler.py"
