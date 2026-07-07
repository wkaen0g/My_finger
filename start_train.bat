@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo ========================================
echo   Gesture Classifier Training
echo   Data: training\data\  -^>  ONNX: models\classifier.onnx
echo ========================================
echo.
python -c "
import logging, sys
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('train')

from pathlib import Path
from microgesture.training.train_classifier import train, NUM_CLASSES
from microgesture.training._hagrid_common import GESTURE_LABELS
from microgesture.training.export_onnx import export_to_onnx
from microgesture.training.model_test import test_model, print_report

data_dir = Path('microgesture/training/data')
test_dir = Path('microgesture/training/test')
model_dir = Path('microgesture/training/models')
onnx_path = Path('microgesture/models/classifier.onnx')

log.info('Labels: %s (%d classes)', GESTURE_LABELS, NUM_CLASSES)
log.info('Training on %s', data_dir.resolve())

# ── Train ──────────────────────────────────────────────────
model = train(data_dir, model_dir, epochs=60, lr=1e-3, test_dir=test_dir)

# ── Test set evaluation ─────────────────────────────────────
if test_dir.exists():
    log.info('=== Test set evaluation (PyTorch) ===')
    metrics = test_model(model_dir / 'best_model.pt', test_dir, model_type='torch')
    print_report(metrics)

# ── Export ONNX ─────────────────────────────────────────────
export_to_onnx(model_dir, onnx_path)
log.info('ONNX model exported to %s', onnx_path.resolve())

# ── Test set evaluation (ONNX) ──────────────────────────────
if test_dir.exists():
    log.info('=== Test set evaluation (ONNX) ===')
    metrics = test_model(onnx_path, test_dir, model_type='onnx')
    print_report(metrics)

log.info('Done.')
"
pause
