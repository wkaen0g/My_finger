@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo ========================================
echo   SINGLE_FINGER Data Collector
echo   Output: training\data\
echo ========================================
echo.
if "%1"=="" (
    python -m microgesture.training.guided_collector --data-dir microgesture\training\data --gestures SINGLE_FINGER --frames 2000
) else (
    python -m microgesture.training.guided_collector --data-dir microgesture\training\data --gestures SINGLE_FINGER --frames %1
)
pause
