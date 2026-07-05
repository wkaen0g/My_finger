@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m microgesture.training.guided_collector
pause
