"""Tests for RuleEngine gesture classification."""
import numpy as np
import pytest
from microgesture.pipeline.gesture_engine import RuleEngine, Gesture, GestureResult


def _make_landmarks(tip_mcp_dists):
    """Build 21x3 landmarks with given tip-MCP distances per finger.

    tip_mcp_dists: list of 5 floats for thumb..pinky
    """
    TIPS = (4, 8, 12, 16, 20)
    MCPS = (2, 5, 9, 13, 17)
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.5, 0.8, 0]  # wrist
    for i, (tip, mcp, d) in enumerate(zip(TIPS, MCPS, tip_mcp_dists)):
        lm[tip] = [0.3 + i * 0.1, 0.3, 0]
        lm[mcp] = [0.3 + i * 0.1, 0.3 + d, 0]
    return lm


class TestRuleEngine:
    def setup_method(self):
        self.engine = RuleEngine()

    def test_fist_all_curled(self):
        """All 5 fingers curled → FIST"""
        lm = _make_landmarks([0.05, 0.05, 0.05, 0.05, 0.05])
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.FIST, f"Expected FIST, got {r.gesture.name}"
        assert r.confidence == 0.9

    def test_fist_four_curled(self):
        """4 fingers curled, 1 semi → still FIST"""
        lm = _make_landmarks([0.05, 0.05, 0.05, 0.20, 0.05])
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.FIST

    def test_palm_open_all_extended(self):
        """All 5 fingers extended → PALM_OPEN"""
        lm = _make_landmarks([0.30, 0.30, 0.30, 0.30, 0.30])
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.PALM_OPEN
        assert r.confidence == 0.7

    def test_two_finger(self):
        """Index+middle extended, ring+pinky curled → TWO_FINGER"""
        lm = _make_landmarks([0.08, 0.30, 0.30, 0.05, 0.05])
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.TWO_FINGER
        assert r.confidence == 0.9

    def test_pinch_thumb_index_close(self):
        """Thumb and index tips close, other fingers relaxed → PINCH"""
        lm = _make_landmarks([0.03, 0.08, 0.25, 0.25, 0.25])
        # Move thumb tip very close to index tip
        lm[4] = lm[8] + np.array([0.015, 0.015, 0], dtype=np.float32)
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.PINCH

    def test_fist_not_pinch(self):
        """FIST must NOT be misclassified as PINCH (regression test)"""
        lm = _make_landmarks([0.08, 0.04, 0.04, 0.04, 0.04])
        # Thumb close to index (as in a fist)
        lm[4] = lm[8] + np.array([0.02, 0.01, 0], dtype=np.float32)
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.FIST, \
            f"FIST misclassified as {r.gesture.name} (bug: PINCH before FIST?)"

    def test_two_finger_not_pinch(self):
        """Peace sign with thumb near index must NOT be PINCH"""
        lm = _make_landmarks([0.04, 0.30, 0.30, 0.05, 0.05])
        lm[4] = lm[8] + np.array([0.02, 0.01, 0], dtype=np.float32)
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.TWO_FINGER, \
            f"TWO_FINGER misclassified as {r.gesture.name}"

    def test_fallback_palm_open(self):
        """Unclear posture → fallback PALM_OPEN with low confidence.
        Fingers are semi-curled (between open=0.25 and fist=0.12),
        NOT pinching (thumb and index tips far apart)."""
        lm = _make_landmarks([0.20, 0.20, 0.20, 0.20, 0.20])  # all "semi"
        # Ensure thumb and index tips are FAR apart (no pinch)
        lm[4] = [0.2, 0.3, 0]
        lm[8] = [0.6, 0.3, 0]  # far from thumb
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.PALM_OPEN
        assert r.confidence == 0.3

    def test_returns_gesture_result(self):
        """classify returns GestureResult with landmarks"""
        lm = _make_landmarks([0.30, 0.30, 0.30, 0.30, 0.30])
        r = self.engine.classify(lm)
        assert isinstance(r, GestureResult)
        assert r.landmarks is lm

    def test_single_finger(self):
        """Index extended, other 4 curled → SINGLE_FINGER"""
        lm = _make_landmarks([0.05, 0.30, 0.05, 0.05, 0.05])
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.SINGLE_FINGER, \
            f"Expected SINGLE_FINGER, got {r.gesture.name}"
        assert r.confidence == 0.9

    def test_two_finger_not_single_finger(self):
        """Index+middle extended → TWO_FINGER, not SINGLE_FINGER"""
        lm = _make_landmarks([0.05, 0.30, 0.30, 0.05, 0.05])
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.TWO_FINGER, \
            f"Expected TWO_FINGER, got {r.gesture.name}"
