"""Tests for AirTapDetector integral tap detection."""
import numpy as np
import pytest
from microgesture.pipeline.air_tap import AirTapDetector, TapResult


def _make_landmarks(ratio):
    """Build landmarks with a specific bend ratio for the index finger.

    ratio = |TIP(8) - MCP(5)| / |PIP(6) - MCP(5)|
    Straight finger: ratio ≈ 2.5-3.0
    Bent finger: ratio drops
    """
    lm = np.zeros((21, 3), dtype=np.float32)
    # Set reference points
    lm[5] = [0.4, 0.6, 0]   # MCP
    lm[6] = [0.4, 0.4, 0]   # PIP
    # Position TIP to achieve desired ratio
    # ratio = |tip - mcp| / |pip - mcp|
    # pip_mcp_dist = |pip - mcp| = |(0.4,0.4) - (0.4,0.6)| = 0.2
    pip_mcp_dist = 0.2
    tip_y = 0.6 - ratio * pip_mcp_dist
    lm[8] = [0.4, tip_y, 0]  # TIP
    return lm


class TestAirTapDetector:
    def setup_method(self):
        self.detector = AirTapDetector(
            tap_threshold=0.3,
            min_bend=0.15,
            suppress_threshold=0.1,
            bend_timeout=12,
            rebound_timeout=8,
            cooldown_frames=8,
        )

    def test_initial_state(self):
        """First frame returns no event and no suppression."""
        lm = _make_landmarks(2.5)  # straight finger
        r = self.detector.update(lm)
        assert isinstance(r, TapResult)
        assert r.event is None
        assert r.suppress_cursor is False

    def test_straight_finger_no_tap(self):
        """Constant straight finger → no tap."""
        self.detector.update(_make_landmarks(2.5))
        for _ in range(50):
            r = self.detector.update(_make_landmarks(2.5))
        assert r.event is None

    def test_bend_and_rebound_triggers_tap(self):
        """Rapid bend then rebound → tap detected."""
        self.detector.update(_make_landmarks(2.5))  # init

        # Bend phase: ratio decreases rapidly
        ratios = [2.5, 2.2, 1.8, 1.4, 1.1, 0.9, 0.8]
        for ratio in ratios:
            r = self.detector.update(_make_landmarks(ratio))
            assert r.event is None, f"Tap should not fire during bend at ratio={ratio}"

        # Rebound phase: ratio increases back
        rebound_ratios = [1.0, 1.3, 1.6, 2.0, 2.3, 2.6]
        tap_fired = False
        for ratio in rebound_ratios:
            r = self.detector.update(_make_landmarks(ratio))
            if r.event is not None:
                tap_fired = True
                break

        assert tap_fired, "Tap should have fired during rebound"

    def test_shallow_bend_no_tap(self):
        """Bend that is too shallow (< min_bend) → no tap."""
        self.detector.update(_make_landmarks(2.5))  # init

        # Very shallow bend
        ratios = [2.5, 2.4, 2.35, 2.3, 2.4, 2.5, 2.6]
        for ratio in ratios:
            r = self.detector.update(_make_landmarks(ratio))
        assert r.event is None, "Shallow bend should not trigger tap"

    def test_suppress_cursor_on_fast_bend(self):
        """Large dratio → suppress_cursor=True."""
        self.detector.update(_make_landmarks(2.5))  # init
        r = self.detector.update(_make_landmarks(2.0))  # big drop
        assert r.suppress_cursor, \
            f"dratio=0.5 > 0.1 should suppress cursor, got suppress={r.suppress_cursor}"

    def test_cooldown_blocks_rapid_taps(self):
        """After a tap fires, cooldown prevents immediate re-trigger."""
        self.detector.update(_make_landmarks(2.5))

        # Fire first tap
        for ratio in [2.2, 1.8, 1.4, 1.0, 0.8]:
            self.detector.update(_make_landmarks(ratio))
        for ratio in [1.0, 1.3, 1.6, 2.0, 2.3, 2.6]:
            r = self.detector.update(_make_landmarks(ratio))
            if r.event:
                break

        # Immediate second tap attempt
        second_tap = False
        for ratio in [2.2, 1.8, 1.2, 0.8, 1.0, 1.3, 1.6, 2.0, 2.3]:
            r = self.detector.update(_make_landmarks(ratio))
            if r.event:
                second_tap = True
                break

        assert not second_tap, "Cooldown should block rapid re-tap"
