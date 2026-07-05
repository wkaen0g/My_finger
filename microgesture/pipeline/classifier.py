"""ONNX Runtime classifier implementing GestureRecognizer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from microgesture.recognition.base import (GestureRecognizer, RecognitionResult,
                                           extract_features)

logger = logging.getLogger(__name__)

from microgesture.training._hagrid_common import GESTURE_LABELS as _LABELS


class ONNXClassifier(GestureRecognizer):
    """MLP gesture classifier backed by ONNX Runtime."""

    def __init__(self, model_path: str | Path):
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(model_path))
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def predict(self, landmarks: np.ndarray) -> RecognitionResult:
        features = extract_features(landmarks)
        features_batch = np.expand_dims(features.astype(np.float32), axis=0)
        logits = self._session.run([self._output_name],
                                   {self._input_name: features_batch})[0][0]

        # Softmax
        max_logit = np.max(logits)
        exps = np.exp(logits - max_logit)
        probs = exps / exps.sum()
        best = int(np.argmax(probs))
        return RecognitionResult(
            label=_LABELS[best],
            confidence=float(probs[best]),
            features=features,
        )

    def close(self) -> None:
        pass  # ONNX session is cleaned up by GC
