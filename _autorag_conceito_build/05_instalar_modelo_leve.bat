@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Instalar-Opcional.ps1" -Grupo local
if errorlevel 1 (pause & exit /b 1)
pause
