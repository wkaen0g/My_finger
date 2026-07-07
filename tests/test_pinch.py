"""Tests for PinchDetector hysteresis state machine."""
import numpy as np
import pytest
from microgesture.pipeline.pinch import PinchDetector, PinchState, PinchEvent


def _make_landmarks(pinch_dist, hand_scale=0.3):
    """Build landmarks with controlled pinch distance.

    pinch_dist = |thumb_tip - index_tip|
    hand_scale = |wrist - middle_mcp|
    normalized = pinch_dist / hand_scale
    """
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.5, 0.8, 0]    # wrist
    lm[9] = [0.5, 0.5, 0]    # middle MCP → hand_scale = 0.3
    lm[4] = [0.4, 0.3, 0]    # thumb tip
    lm[8] = [0.4 + pinch_dist, 0.3, 0]  # index tip
    return lm


class TestPinchDetector:
    def setup_method(self):
        self.detector = PinchDetector(
            pinch_threshold_ratio=0.35,
            release_threshold_ratio=0.55,
        )

    def test_initial_state_open(self):
        """PinchDetector starts in OPEN state."""
        assert self.detector.is_pinching is False

    def test_open_stays_open(self):
        """Wide pinch distance → stays OPEN, no event."""
        lm = _make_landmarks(pinch_dist=0.2)  # norm = 0.2/0.3 = 0.67 > 0.55
        r = self.detector.update(lm)
        assert r is None
        assert not self.detector.is_pinching

    def test_close_triggers_pinch_after_debounce(self):
        """Close distance for 3+ frames → PINCHING event."""
        lm = _make_landmarks(pinch_dist=0.06)  # norm = 0.06/0.3 = 0.20 < 0.35
        for _ in range(3):
            r = self.detector.update(lm)
        assert r is not None
        assert r.state == PinchState.PINCHING
        assert self.detector.is_pinching

    def test_close_needs_three_frames_debounce(self):
        """Single close frame → no event (debounce)."""
        lm_close = _make_landmarks(pinch_dist=0.06)
        r1 = self.detector.update(lm_close)
        assert r1 is None, "First close frame should be debounced"
        r2 = self.detector.update(lm_close)  # 2nd frame
        assert r2 is None, "Second close frame should be debounced"

    def test_release_after_pinch(self):
        """Open after pinching → OPEN event."""
        lm_close = _make_landmarks(pinch_dist=0.06)
        lm_open = _make_landmarks(pinch_dist=0.2)

        # Pinch
        for _ in range(3):
            self.detector.update(lm_close)
        assert self.detector.is_pinching

        # Release
        for _ in range(3):
            r = self.detector.update(lm_open)
        assert r is not None
        assert r.state == PinchState.OPEN
        assert not self.detector.is_pinching

    def test_noise_filtered(self):
        """Brief noise → no state change."""
        lm_close = _make_landmarks(pinch_dist=0.06)
        lm_open = _make_landmarks(pinch_dist=0.2)

        self.detector.update(lm_close)  # 1 frame close
        self.detector.update(lm_open)   # back to open
        r = self.detector.update(lm_open)
        assert r is None
        assert not self.detector.is_pinching
