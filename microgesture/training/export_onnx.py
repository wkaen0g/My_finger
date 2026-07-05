"""Export trained GestureMLP to ONNX format."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from .train_classifier import GestureMLP, _GESTURES

logger = logging.getLogger(__name__)


def export_to_onnx(
    model_dir: str | Path,
    output_path: str | Path,
    *,
    hidden: int = 128,
    dropout: float = 0.3,
) -> Path:
    """Load best_model.pt, export to ONNX, return output path."""

    model_dir = Path(model_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = GestureMLP(input_dim=70, hidden=hidden, dropout=dropout)
    state = torch.load(model_dir / "best_model.pt", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 70)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    logger.info("Exported ONNX model → %s (labels: %s)", output_path, _GESTURES)
    return output_path
