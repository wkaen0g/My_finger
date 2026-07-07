@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo ========================================
echo   Gesture Test Data Collector
echo   5 gestures, 1000 frames each
echo   Output: training\data_test\
echo ========================================
echo.
python -m microgesture.training.guided_collector --data-dir microgesture\training\data_test --frames 1000
pause
