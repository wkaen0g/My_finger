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

    _FINGER_NAMES = ("拇指", "食指", "中指", "无名指", "小指")

    def __init__(self, tip_mcp_open_threshold: float = 0.25,
                 tip_mcp_fist_threshold: float = 0.12,
                 pinch_threshold_ratio: float = 0.35):
        self.open_threshold = tip_mcp_open_threshold
        self.fist_threshold = tip_mcp_fist_threshold
        self.pinch_ratio = pinch_threshold_ratio
        self._frame = 0

    def classify(self, landmarks: np.ndarray) -> GestureResult:
        """Classify hand landmarks into a gesture category."""
        self._frame += 1
        distances = []
        for tip_idx, mcp_idx in zip(TIP, MCP):
            d = np.linalg.norm(landmarks[tip_idx] - landmarks[mcp_idx])
            distances.append(d)

        open_count = sum(1 for d in distances if d > self.open_threshold)
        closed_count = sum(1 for d in distances if d < self.fist_threshold)

        # ── per-finger debug (every 5th frame) ──
        if self._frame % 5 == 0:
            parts = ", ".join(
                f"{n}={distances[i]:.3f}"
                f"({'伸' if d > self.open_threshold else '屈' if d < self.fist_threshold else '半'})"
                for i, (n, d) in enumerate(zip(self._FINGER_NAMES, distances))
            )
            logger.debug("指尖-MCP距离: %s | 伸=%d 屈=%d", parts, open_count, closed_count)

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
        pinch_norm = pinch_dist / hand_scale if hand_scale > 0 else 1.0
        if self._frame % 5 == 0:
            logger.debug("捏合: dist=%.3f scale=%.3f norm=%.3f thresh=%.2f",
                         pinch_dist, hand_scale, pinch_norm, self.pinch_ratio)
        if pinch_norm < self.pinch_ratio:
            return GestureResult(Gesture.PINCH, landmarks, 1.0)

        # Default to nearest match or palm_open as safe fallback
        if self._frame % 5 == 0:
            logger.debug("手势判定: PALM_OPEN (fallback, conf=0.5)")
        return GestureResult(Gesture.PALM_OPEN, landmarks, 0.5)
