"""Cursor movement controller with 1-euro filter smoothing."""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class OneEuroFilter:
    """1€ filter for smoothing noisy signals with adaptive cutoff."""

    def __init__(self, beta: float = 0.007, fcmin: float = 1.0,
                 min_cutoff: float = 1.0, fps: float = 30.0):
        self.beta = beta
        self.fcmin = fcmin
        self.min_cutoff = min_cutoff
        self.fps = fps
        self._x_prev: Optional[float] = None
        self._dx_prev: Optional[float] = None
        self._initialized = False

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / self.fps
        return 1.0 / (1.0 + tau / te)

    def filter(self, value: float) -> float:
        if not self._initialized:
            self._x_prev = value
            self._dx_prev = 0.0
            self._initialized = True
            return value

        # Estimate derivative
        dx = (value - self._x_prev) * self.fps

        # Smooth derivative
        edx = self._alpha(self.min_cutoff) * dx + (1 - self._alpha(self.min_cutoff)) * self._dx_prev

        # Adaptive cutoff based on speed
        cutoff = self.fcmin + self.beta * abs(edx)

        # Smooth value
        x_hat = self._alpha(cutoff) * value + (1 - self._alpha(cutoff)) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = edx

        return x_hat

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None
        self._initialized = False


class CursorController:
    """Maps fingertip displacement to cursor movement with smoothing."""

    def __init__(self, sensitivity: float = 1.5, deadzone: float = 0.003,
                 beta: float = 0.007, fcmin: float = 1.0, min_cutoff: float = 1.0,
                 fps: float = 30.0):
        self.sensitivity = sensitivity
        self.deadzone = deadzone
        self._filter_x = OneEuroFilter(beta, fcmin, min_cutoff, fps)
        self._filter_y = OneEuroFilter(beta, fcmin, min_cutoff, fps)
        self._prev_raw: Optional[Tuple[float, float]] = None
        self._frozen = False

    def freeze(self) -> None:
        self._frozen = True
        self._prev_raw = None
        self._filter_x.reset()
        self._filter_y.reset()

    def unfreeze(self) -> None:
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def update(self, x: float, y: float) -> Tuple[float, float]:
        """Returns (dx, dy) cursor delta in pixels, or (0,0) when frozen."""
        if self._frozen:
            return (0.0, 0.0)

        sx = self._filter_x.filter(x)
        sy = self._filter_y.filter(y)

        if self._prev_raw is None:
            self._prev_raw = (sx, sy)
            return (0.0, 0.0)

        raw_dx = (sx - self._prev_raw[0]) * self.sensitivity * 1920
        raw_dy = (sy - self._prev_raw[1]) * self.sensitivity * 1080

        # Dead zone: ignore sub-pixel jitter
        dz = self.deadzone * 1920
        if abs(raw_dx) < dz:
            raw_dx = 0.0
        if abs(raw_dy) < dz:
            raw_dy = 0.0

        self._prev_raw = (sx, sy)
        return (raw_dx, raw_dy)
