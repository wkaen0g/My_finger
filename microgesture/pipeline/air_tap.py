"""Air tap detection via index finger distance ratio + integral phasing.

ratio = |TIP(8) - MCP(5)| / |PIP(6) - MCP(5)|

Straight finger: ratio ≈ 2.5–3.0
Bending finger:   ratio drops → Δratio negative
Rebounding:       ratio rises → Δratio positive

Tap detection uses a two-phase integrator:
  BEND phase    — accumulate |Δratio| while Δratio < 0
  REBOUND phase — accumulate  Δratio  while Δratio > 0
  A tap fires when both phases cross their cumulative thresholds
  within timeout windows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Index-finger keypoints
MCP_IDX = 5   # knuckle
PIP_IDX = 6   # middle joint
TIP_IDX = 8   # fingertip


class _Phase(Enum):
    IDLE = auto()
    BENDING = auto()
    REBOUNDING = auto()


@dataclass
class TapEvent:
    timestamp: float


@dataclass
class TapResult:
    event: Optional[TapEvent]
    suppress_cursor: bool
    ratio: float
    dratio: float


class AirTapDetector:
    """Integral tap detector with separate bend / rebound accumulation."""

    _DEBUG_INTERVAL = 300  # log phase transitions every N frames max

    def __init__(
        self,
        *,
        tap_threshold: float = 0.3,          # cumulative rebound sum to fire
        min_bend: float = 0.15,              # minimum bend sum for a valid wind-up
        suppress_threshold: float = 0.1,     # |Δratio| above which cursor is suppressed
        bend_timeout: int = 12,              # max frames in BEND phase before reset
        rebound_timeout: int = 8,            # max frames in REBOUND phase before reset
        cooldown_frames: int = 8,            # frames to ignore after a tap
    ):
        self.tap_threshold = tap_threshold
        self.min_bend = min_bend
        self.suppress_threshold = suppress_threshold
        self.bend_timeout = bend_timeout
        self.rebound_timeout = rebound_timeout
        self.cooldown_frames = cooldown_frames

        self._phase = _Phase.IDLE
        self._bend_sum = 0.0
        self._rebound_sum = 0.0
        self._phase_frames = 0
        self._tap_cooldown = 0
        self._prev_ratio: Optional[float] = None
        self._frame = 0

    def _debug(self, msg: str, *args) -> None:
        """Rate-limited debug log (every N frames)."""
        if self._frame % self._DEBUG_INTERVAL == 0:
            logger.debug(msg, *args)

    # ── public API ──────────────────────────────────────────────────────

    def update(self, landmarks: np.ndarray) -> TapResult:
        self._frame += 1
        mcp = landmarks[MCP_IDX]
        pip = landmarks[PIP_IDX]
        tip = landmarks[TIP_IDX]

        ratio = np.linalg.norm(tip - mcp) / (np.linalg.norm(pip - mcp) + 1e-6)

        if self._prev_ratio is None:
            self._prev_ratio = ratio
            return TapResult(event=None, suppress_cursor=False,
                             ratio=ratio, dratio=0.0)

        dratio = ratio - self._prev_ratio
        self._prev_ratio = ratio

        if self._tap_cooldown > 0:
            self._tap_cooldown -= 1

        suppress = abs(dratio) > self.suppress_threshold
        tap_event = self._run_phase_machine(dratio)

        return TapResult(event=tap_event, suppress_cursor=suppress,
                         ratio=ratio, dratio=dratio)

    # ── phase machine ───────────────────────────────────────────────────

    def _run_phase_machine(self, dratio: float) -> Optional[TapEvent]:
        if self._tap_cooldown > 0:
            self._phase = _Phase.IDLE
            return None

        if self._phase == _Phase.IDLE:
            return self._handle_idle(dratio)
        elif self._phase == _Phase.BENDING:
            return self._handle_bending(dratio)
        else:
            return self._handle_rebounding(dratio)

    def _handle_idle(self, dratio: float) -> Optional[TapEvent]:
        if dratio < -self.suppress_threshold:
            self._phase = _Phase.BENDING
            self._bend_sum = abs(dratio)
            self._phase_frames = 1
            self._debug("点按: 进入BEND相 sum=%.3f", self._bend_sum)
        return None

    def _handle_bending(self, dratio: float) -> Optional[TapEvent]:
        self._phase_frames += 1

        # Timeout — abandoned wind-up
        if self._phase_frames > self.bend_timeout:
            self._debug("点按: BEND超时 frames=%d sum=%.3f",
                         self._phase_frames, self._bend_sum)
            self._phase = _Phase.IDLE
            return None

        # Still bending — accumulate
        if dratio <= 0:
            self._bend_sum += abs(dratio)
            return None

        # Crossed zero → attempt transition to rebound
        if self._bend_sum < self.min_bend:
            self._debug("点按: BEND太浅 sum=%.3f < min=%.2f",
                         self._bend_sum, self.min_bend)
            self._phase = _Phase.IDLE
            return None

        self._phase = _Phase.REBOUNDING
        self._rebound_sum = dratio
        self._phase_frames = 1
            self._debug("点按: 进入REBOUND相 bend_sum=%.3f", self._bend_sum)
        return None

    def _handle_rebounding(self, dratio: float) -> Optional[TapEvent]:
        self._phase_frames += 1

        # Timeout — rebound took too long
        if self._phase_frames > self.rebound_timeout:
            self._debug("点按: REBOUND超时 frames=%d sum=%.3f thresh=%.2f",
                         self._phase_frames, self._rebound_sum, self.tap_threshold)
            self._phase = _Phase.IDLE
            return None

        # Dipped back negative — false alarm, discard
        if dratio < -self.suppress_threshold:
            self._debug("点按: REBOUND回退 取消")
            self._phase = _Phase.IDLE
            return None

        # Still rebounding — accumulate
        if dratio >= 0:
            self._rebound_sum += dratio
            if self._rebound_sum >= self.tap_threshold:
                logger.info("空中点按! bend=%.3f rebound=%.3f",
                             self._bend_sum, self._rebound_sum)
                self._phase = _Phase.IDLE
                self._tap_cooldown = self.cooldown_frames
                return TapEvent(timestamp=0.0)

        return None
