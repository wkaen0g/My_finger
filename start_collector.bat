@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo ========================================
echo   手势训练数据采集工具
echo   采集 5 类手势，增量追加（不覆盖旧数据）
echo   输出: microgesture\training\data\
echo   用法: start_collector.bat [帧数]
echo         start_collector.bat 2000  → 每类追加2000帧
echo         start_collector.bat       → 每类追加1000帧
echo ========================================
echo.
if "%1"=="" (
    python -m microgesture.training.guided_collector --data-dir microgesture\training\data --frames 1000
) else (
    python -m microgesture.training.guided_collector --data-dir microgesture\training\data --frames %1
)
pause
