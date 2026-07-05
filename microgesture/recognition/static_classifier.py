"""Static (rule-engine) recognizer implementing GestureRecognizer."""

from __future__ import annotations

import logging

import numpy as np

from .base import GestureRecognizer, RecognitionResult, extract_features

logger = logging.getLogger(__name__)


class StaticClassifier(GestureRecognizer):
    """Wraps the Phase-1 RuleEngine behind the GestureRecognizer interface."""

    def __init__(self, rule_engine):
        self._engine = rule_engine

    def predict(self, landmarks: np.ndarray) -> RecognitionResult:
        features = extract_features(landmarks)
        gr = self._engine.classify(landmarks)
        return RecognitionResult(
            label=gr.gesture.name,
            confidence=gr.confidence,
            features=features,
        )

    def close(self) -> None:
        pass
