"""Air tap detection via index finger distance ratio + cursor suppression."""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Keypoint indices for index finger
MCP_IDX = 5   # metacarpophalangeal (knuckle)
PIP_IDX = 6   # proximal interphalangeal (middle joint)
TIP_IDX = 8   # fingertip


@dataclass
class TapEvent:
    timestamp: float


@dataclass
class TapResult:
    """Output of one frame of tap analysis."""
    event: Optional[TapEvent]  # tap detected this frame
    suppress_cursor: bool       # cursor should be suppressed this frame
    ratio: float                # current |TIP-MCP| / |PIP-MCP| (diagnostic)
    dratio: float              # frame-to-frame ratio change (diagnostic)


class AirTapDetector:
    """Detects finger taps using index finger distance ratio.

    ratio = |TIP - MCP| / |PIP - MCP|

    When finger is straight: ratio ≈ 2.5–3.0
    When PIP bends (tap):   ratio drops → Δratio negative
    When finger rebounds:   ratio rises → Δratio positive

    Tap = large negative Δratio pulse followed by large positive within rebound window.
    Cursor suppression activates whenever |Δratio| exceeds suppress_threshold.
    """

    def __init__(self, tap_threshold: float = 0.5,
                 suppress_threshold: float = 0.2,
                 rebound_frames: int = 5,
                 cooldown_frames: int = 8):
        self.tap_threshold = tap_threshold
        self.suppress_threshold = suppress_threshold
        self.rebound_frames = rebound_frames
        self.cooldown_frames = cooldown_frames
        self._dratio_history: deque[float] = deque(maxlen=10)
        self._prev_ratio: Optional[float] = None
        self._tap_cooldown = 0

    def update(self, landmarks: np.ndarray) -> TapResult:
        """Process landmarks. Returns TapResult with event + suppression flag."""
        mcp = landmarks[MCP_IDX]
        pip = landmarks[PIP_IDX]
        tip = landmarks[TIP_IDX]

        ratio = (np.linalg.norm(tip - mcp)
                 / (np.linalg.norm(pip - mcp) + 1e-6))

        if self._prev_ratio is None:
            self._prev_ratio = ratio
            return TapResult(event=None, suppress_cursor=False,
                             ratio=ratio, dratio=0.0)

        dratio = ratio - self._prev_ratio
        self._prev_ratio = ratio
        self._dratio_history.append(dratio)

        # Cursor suppression: any significant finger bending blocks movement
        suppress = abs(dratio) > self.suppress_threshold

        # Tap detection: negative pulse (bend) → positive pulse (rebound)
        tap_event = None
        if self._tap_cooldown > 0:
            self._tap_cooldown -= 1
        elif len(self._dratio_history) >= 3:
            a_prev = self._dratio_history[-2]
            a_curr = self._dratio_history[-1]
            if a_prev < -self.tap_threshold and a_curr > self.tap_threshold:
                self._tap_cooldown = self.cooldown_frames
                logger.debug("Air tap detected (dratio: %.3f → %.3f)", a_prev, a_curr)
                tap_event = TapEvent(timestamp=0.0)

        return TapResult(event=tap_event, suppress_cursor=suppress,
                         ratio=ratio, dratio=dratio)
