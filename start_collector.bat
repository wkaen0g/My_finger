@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo ========================================
echo   Gesture Training Data Collector
echo   5 gestures, incremental append
echo   Output: training\data\
echo   Usage: start_collector.bat [frames]
echo          start_collector.bat 2000
echo          start_collector.bat
echo ========================================
echo.
if "%1"=="" (
    python -m microgesture.training.guided_collector --data-dir microgesture\training\data --frames 1000
) else (
    python -m microgesture.training.guided_collector --data-dir microgesture\training\data --frames %1
)
pause
