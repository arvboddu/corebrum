@echo off
echo Starting Corebrum HUD...
cd /d "%~dp0"
node_modules\electron\dist\electron.exe .
echo.
echo Press any key to exit...
pause > nul
