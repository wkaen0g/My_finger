@echo off
cd /d "%~dp0"
start code "%~dp0"
call .venv\Scripts\activate.bat
python -m microgesture.main
pause
