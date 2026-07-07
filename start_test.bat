@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo ========================================
echo   Model Test — ONNX ^& PyTorch
echo   Test set: training\data_test\
echo ========================================
echo.
echo ── ONNX ──────────────────────────
python -m microgesture.training.model_test --model onnx --data-dir microgesture\training\data_test
echo.
echo ── PyTorch ───────────────────────
python -m microgesture.training.model_test --model torch --data-dir microgesture\training\data_test
pause
