@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo ========================================
echo   手势测试集采集工具
echo   采集 5 类手势，每类 500 帧
echo   输出: microgesture\training\data_test\
echo ========================================
echo.
python -m microgesture.training.guided_collector --data-dir microgesture\training\data_test
pause
