@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Configurar-Ollama.ps1" %*
if errorlevel 1 (pause & exit /b 1)
pause
