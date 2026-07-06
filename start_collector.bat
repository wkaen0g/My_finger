@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo ========================================
echo   手势训练数据采集工具
echo   采集 5 类手势，每类 500 帧
echo   输出: microgesture\training\data\
echo   增加数据量: start_collector.bat 1000
echo ========================================
echo.
if "%1"=="" (
    python -m microgesture.training.guided_collector --data-dir microgesture\training\data
) else (
    python -m microgesture.training.guided_collector --data-dir microgesture\training\data --frames %1
)
pause
