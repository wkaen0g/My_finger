"""Two-finger vertical scroll detection."""

import logging
from typing import Optional

import numpy as np

from .cursor import OneEuroFilter

logger = logging.getLogger(__name__)


class ScrollDetector:
    """Detects two-finger scroll gesture from index+middle fingertip midpoint."""

    def __init__(self, screen_height: int = 1080, sensitivity: float = 40.0,
                 beta: float = 0.007, fcmin: float = 1.0, min_cutoff: float = 1.0,
                 fps: float = 30.0):
        self._sh = screen_height
        self.sensitivity = sensitivity
        self._filter = OneEuroFilter(beta, fcmin, min_cutoff, fps)
        self._prev_y: Optional[float] = None
        self._active = False

    def start(self) -> None:
        self._active = True
        self._prev_y = None
        self._filter.reset()

    def stop(self) -> None:
        self._active = False
        self._prev_y = None

    @property
    def is_active(self) -> bool:
        return self._active

    def update(self, index_tip: np.ndarray, middle_tip: np.ndarray) -> float:
        """Returns scroll delta (positive=up, negative=down), 0 if inactive."""
        if not self._active:
            return 0.0

        mid_y = (index_tip[1] + middle_tip[1]) / 2.0
        sy = self._filter.filter(mid_y)

        if self._prev_y is None:
            self._prev_y = sy
            return 0.0

        # y increases downward in image coords, scroll up = positive
        dy = (self._prev_y - sy) * self.sensitivity * self._sh
        self._prev_y = sy
        return dy
