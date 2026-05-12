"""Rule-based gesture classification using hand landmark geometry."""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Gesture(Enum):
    NO_HAND = auto()
    PALM_OPEN = auto()
    FIST = auto()
    TWO_FINGER = auto()
    PINCH = auto()


@dataclass
class GestureResult:
    gesture: Gesture
    landmarks: np.ndarray  # (21, 3) normalized coords
    confidence: float

# MediaPipe hand landmark indices
TIP = (4, 8, 12, 16, 20)     # thumb, index, middle, ring, pinky tips
MCP = (2, 5, 9, 13, 17)      # corresponding MCP joints
INDEX_TIP = 8
MIDDLE_TIP = 12
THUMB_TIP = 4
WRIST = 0
MIDDLE_MCP = 9


class RuleEngine:
    """Geometry-based gesture classifier."""

    def __init__(self, tip_mcp_open_threshold: float = 0.25,
                 tip_mcp_fist_threshold: float = 0.12,
                 pinch_threshold_ratio: float = 0.35):
        self.open_threshold = tip_mcp_open_threshold
        self.fist_threshold = tip_mcp_fist_threshold
        self.pinch_ratio = pinch_threshold_ratio

    def classify(self, landmarks: np.ndarray) -> GestureResult:
        """Classify hand landmarks into a gesture category."""
        distances = []
        for tip_idx, mcp_idx in zip(TIP, MCP):
            d = np.linalg.norm(landmarks[tip_idx] - landmarks[mcp_idx])
            distances.append(d)

        open_count = sum(1 for d in distances if d > self.open_threshold)
        closed_count = sum(1 for d in distances if d < self.fist_threshold)

        # Palm open: all 5 tips extended
        if open_count == 5:
            return GestureResult(Gesture.PALM_OPEN, landmarks, 1.0)

        # Fist: all 5 tips curled
        if closed_count == 5:
            return GestureResult(Gesture.FIST, landmarks, 1.0)

        # Two-finger: index + middle extended, ring + pinky curled
        if (distances[1] > self.open_threshold and distances[2] > self.open_threshold
                and distances[3] < self.fist_threshold and distances[4] < self.fist_threshold):
            return GestureResult(Gesture.TWO_FINGER, landmarks, 1.0)

        # Pinch: thumb tip close to index tip relative to hand size
        pinch_dist = np.linalg.norm(landmarks[THUMB_TIP] - landmarks[INDEX_TIP])
        hand_scale = np.linalg.norm(landmarks[WRIST] - landmarks[MIDDLE_MCP])
        if hand_scale > 0 and pinch_dist / hand_scale < self.pinch_ratio:
            return GestureResult(Gesture.PINCH, landmarks, 1.0)

        # Default to nearest match or palm_open as safe fallback
        return GestureResult(Gesture.PALM_OPEN, landmarks, 0.5)
