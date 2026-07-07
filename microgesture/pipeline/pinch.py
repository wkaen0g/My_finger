"""Pinch detection state machine."""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)
TRACE = 5  # per-frame detail, below DEBUG (10)


class PinchState(Enum):
    OPEN = auto()
    PINCHING = auto()


@dataclass
class PinchEvent:
    state: PinchState


class PinchDetector:
    """State machine for thumb-index pinch detection with hysteresis."""

    _DEBUG_INTERVAL = 5

    def __init__(self, pinch_threshold_ratio: float = 0.35,
                 release_threshold_ratio: float = 0.55):
        self.start_threshold = pinch_threshold_ratio     # norm < this → close
        self.release_threshold = release_threshold_ratio  # norm > this → open
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

        if self._state == PinchState.OPEN:
            # Start pinch: need norm below start_threshold (tighter)
            is_close = normalized_dist < self.start_threshold
            if is_close:
                self._stable_count += 1
                if self._stable_count >= self._stable_threshold:
                    self._state = PinchState.PINCHING
                    self._stable_count = 0
                    logger.info("捏合开始 (norm=%.3f, start=%.2f)", normalized_dist, self.start_threshold)
                    return PinchEvent(state=PinchState.PINCHING)
            else:
                self._stable_count = 0
        else:
            # Release pinch: need norm above release_threshold (wider gap)
            is_open = normalized_dist > self.release_threshold
            if is_open:
                self._stable_count += 1
                if self._stable_count >= self._stable_threshold:
                    self._state = PinchState.OPEN
                    self._stable_count = 0
                    logger.info("捏合释放 (norm=%.3f, release=%.2f)", normalized_dist, self.release_threshold)
                    return PinchEvent(state=PinchState.OPEN)
            else:
                self._stable_count = 0

        if self._frame % self._DEBUG_INTERVAL == 0:
            logger.log(TRACE, "捏合参数: dist=%.3f scale=%.3f norm=%.3f 开始=%.2f 释放=%.2f 状态=%s 计数=%d",
                         pinch_dist, hand_scale, normalized_dist,
                         self.start_threshold, self.release_threshold,
                         self._state.name, self._stable_count)
        return None
