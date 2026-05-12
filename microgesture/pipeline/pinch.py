"""Pinch detection state machine."""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PinchState(Enum):
    OPEN = auto()
    PINCHING = auto()


@dataclass
class PinchEvent:
    state: PinchState


class PinchDetector:
    """State machine for thumb-index pinch detection."""

    def __init__(self, pinch_threshold_ratio: float = 0.35):
        self.threshold_ratio = pinch_threshold_ratio
        self._state = PinchState.OPEN
        self._stable_count = 0
        self._stable_threshold = 3  # frames to confirm state change
        self._frame = 0

    @property
    def is_pinching(self) -> bool:
        return self._state == PinchState.PINCHING

    def update(self, landmarks: np.ndarray) -> Optional[PinchEvent]:
        """Process landmarks, returns event on state change."""
        self._frame += 1
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        pinch_dist = np.linalg.norm(thumb_tip - index_tip)
        hand_scale = np.linalg.norm(landmarks[0] - landmarks[9])
        normalized_dist = pinch_dist / hand_scale if hand_scale > 0 else 1.0

        is_close = normalized_dist < self.threshold_ratio

        if self._state == PinchState.OPEN and is_close:
            self._stable_count += 1
            if self._stable_count >= self._stable_threshold:
                self._state = PinchState.PINCHING
                self._stable_count = 0
                logger.info("捏合开始 (norm=%.3f, thresh=%.2f)", normalized_dist, self.threshold_ratio)
                return PinchEvent(state=PinchState.PINCHING)
        elif self._state == PinchState.PINCHING and not is_close:
            self._stable_count += 1
            if self._stable_count >= self._stable_threshold:
                self._state = PinchState.OPEN
                self._stable_count = 0
                logger.info("捏合释放 (norm=%.3f, thresh=%.2f)", normalized_dist, self.threshold_ratio)
                return PinchEvent(state=PinchState.OPEN)
        else:
            self._stable_count = 0

        if self._frame % 5 == 0:
            logger.debug("捏合参数: dist=%.3f scale=%.3f norm=%.3f 阈值=%.2f 状态=%s 计数=%d",
                         pinch_dist, hand_scale, normalized_dist, self.threshold_ratio,
                         self._state.name, self._stable_count)
        return None
