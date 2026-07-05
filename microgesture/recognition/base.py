"""Abstract base class for gesture recognizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RecognitionResult:
    """Output from a gesture recognizer."""
    label: str          # e.g. "PALM_OPEN", "FIST", "TWO_FINGER", "PINCH", "NO_HAND"
    confidence: float   # [0, 1]
    features: np.ndarray  # the feature vector used for prediction (70-dim)


class GestureRecognizer(ABC):
    """Interface that all gesture recognizers must implement."""

    @abstractmethod
    def predict(self, landmarks: np.ndarray) -> RecognitionResult:
        """Classify a 21x3 landmark array into a gesture label."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by this recognizer."""
        ...


# ── shared feature extraction ─────────────────────────────────────────────

TIPS = (4, 8, 12, 16, 20)
MCPS = (2, 5, 9, 13, 17)


def extract_features(landmarks: np.ndarray) -> np.ndarray:
    """Build a 70-dim feature vector from 21x3 landmarks.

    63 raw coords + 5 tip-MCP distances + 1 pinch distance + 1 index bend ratio.
    """
    raw = landmarks.astype(np.float32).flatten()

    tip_mcp_dists = [
        np.linalg.norm(landmarks[t] - landmarks[m])
        for t, m in zip(TIPS, MCPS)
    ]
    pinch_dist = np.linalg.norm(landmarks[4] - landmarks[8])
    ratio = np.linalg.norm(landmarks[8] - landmarks[5]) / (
        np.linalg.norm(landmarks[6] - landmarks[5]) + 1e-6
    )

    geo = np.array([*tip_mcp_dists, pinch_dist, ratio], dtype=np.float32)
    return np.concatenate([raw, geo])
