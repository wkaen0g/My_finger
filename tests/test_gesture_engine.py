"""Tests for RuleEngine gesture classification."""
import numpy as np
import pytest
from microgesture.pipeline.gesture_engine import RuleEngine, Gesture, GestureResult


def _make_landmarks(tip_mcp_dists):
    """Build 21x3 landmarks with known hand_scale = 0.5.

    Normalized tip-MCP = raw_d / 0.5.
    With defaults (open_ratio=0.70, fist_ratio=0.45):
      - Extended: raw_d > 0.35
      - Curled:   raw_d < 0.225
      - Semi:      0.225 ≤ raw_d ≤ 0.35

    tip_mcp_dists: list of 5 floats for thumb..pinky
    """
    TIPS = (4, 8, 12, 16, 20)
    MCPS = (2, 5, 9, 13, 17)
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.2, 0.5, 0]   # wrist
    lm[9] = [0.2, 1.0, 0]   # middle MCP → hand_scale = 0.5
    for i, (tip, mcp, d) in enumerate(zip(TIPS, MCPS, tip_mcp_dists)):
        lm[mcp] = [0.1 * i, 1.0, 0]
        lm[tip] = [0.1 * i, 1.0 + d, 0]
    return lm


class TestRuleEngine:
    def setup_method(self):
        self.engine = RuleEngine(tip_mcp_open_threshold=0.70,
                                 tip_mcp_fist_threshold=0.45,
                                 single_mid_max=0.60,
                                 single_ring_max=0.55,
                                 single_idx_ratio=1.3)

    def test_fist_all_curled(self):
        """All 5 fingers equally curled → FIST"""
        lm = _make_landmarks([0.1, 0.1, 0.1, 0.1, 0.1])
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.FIST, f"Expected FIST, got {r.gesture.name}"
        assert r.confidence == 0.9

    def test_fist_four_curled(self):
        """4 fingers curled, 1 semi-ring → still FIST (index NOT dominant)"""
        lm = _make_landmarks([0.1, 0.1, 0.1, 0.35, 0.1])
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.FIST

    def test_fist_not_steal_dominant_index(self):
        """Index clearly dominant (semi while others curled) → NOT FIST.
        This is SINGLE_FINGER mid-tap — index bent but still taller than others."""
        lm = _make_landmarks([0.08, 0.22, 0.08, 0.08, 0.08])
        # index=0.44 normalized (semi), others=0.16 → index > others_max*1.5
        # Move thumb MCP+tip away from index to avoid accidentally triggering PINCH
        lm[2] = [0.3, 1.0, 0]   # thumb MCP
        lm[4] = [0.3, 1.08, 0]  # thumb tip (distance = 0.08 preserved)
        r = self.engine.classify(lm)
        assert r.gesture != Gesture.FIST, \
            f"FIST guard failed: should not steal from SINGLE_FINGER"
        # Falls through to fallback PALM_OPEN (armed tap latch handles dispatch)

    def test_palm_open_all_extended(self):
        """All 5 fingers extended → PALM_OPEN"""
        lm = _make_landmarks([0.5, 0.5, 0.5, 0.5, 0.5])
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.PALM_OPEN
        assert r.confidence == 0.7

    def test_two_finger(self):
        """Index+middle extended, ring+pinky curled → TWO_FINGER"""
        lm = _make_landmarks([0.1, 0.5, 0.5, 0.1, 0.1])
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.TWO_FINGER
        assert r.confidence == 0.9

    def test_pinch_thumb_index_close(self):
        """Thumb and index tips close, other fingers relaxed → PINCH"""
        lm = _make_landmarks([0.1, 0.2, 0.4, 0.4, 0.4])
        # Move thumb tip very close to index tip
        lm[4] = lm[8] + np.array([0.015, 0.015, 0], dtype=np.float32)
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.PINCH

    def test_fist_not_pinch(self):
        """FIST must NOT be misclassified as PINCH (regression test)"""
        lm = _make_landmarks([0.1, 0.1, 0.1, 0.1, 0.1])
        # Thumb close to index (as in a fist)
        lm[4] = lm[8] + np.array([0.02, 0.01, 0], dtype=np.float32)
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.FIST, \
            f"FIST misclassified as {r.gesture.name} (bug: PINCH before FIST?)"

    def test_two_finger_not_pinch(self):
        """Peace sign with thumb near index must NOT be PINCH"""
        lm = _make_landmarks([0.1, 0.5, 0.5, 0.1, 0.1])
        lm[4] = lm[8] + np.array([0.02, 0.01, 0], dtype=np.float32)
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.TWO_FINGER, \
            f"TWO_FINGER misclassified as {r.gesture.name}"

    def test_fallback_palm_open(self):
        """Unclear posture → fallback PALM_OPEN with low confidence.
        Fingers are semi-curled (normalized between 0.45 and 0.70),
        NOT pinching (thumb and index tips far apart)."""
        lm = _make_landmarks([0.3, 0.3, 0.3, 0.3, 0.3])  # all "semi" (0.6)
        # Ensure thumb and index tips are FAR apart (no pinch), while
        # keeping them above their MCPs to preserve raw tip-MCP distances.
        lm[2] = [0.3, 1.0, 0]   # thumb MCP away from index
        lm[4] = [0.3, 1.3, 0]   # thumb tip (d=0.3 preserved)
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.PALM_OPEN
        assert r.confidence == 0.3

    def test_returns_gesture_result(self):
        """classify returns GestureResult with landmarks"""
        lm = _make_landmarks([0.5, 0.5, 0.5, 0.5, 0.5])
        r = self.engine.classify(lm)
        assert isinstance(r, GestureResult)
        assert r.landmarks is lm

    def test_single_finger(self):
        """Index extended, other 4 curled → SINGLE_FINGER"""
        lm = _make_landmarks([0.1, 0.5, 0.1, 0.1, 0.1])
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.SINGLE_FINGER, \
            f"Expected SINGLE_FINGER, got {r.gesture.name}"
        assert r.confidence == 0.9

    def test_two_finger_not_single_finger(self):
        """Index+middle extended → TWO_FINGER, not SINGLE_FINGER"""
        lm = _make_landmarks([0.1, 0.5, 0.5, 0.1, 0.1])
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.TWO_FINGER, \
            f"Expected TWO_FINGER, got {r.gesture.name}"

    def test_single_finger_thumb_ignored(self):
        """Thumb semi-open does NOT break SINGLE_FINGER (independent movement)"""
        lm = _make_landmarks([0.35, 0.5, 0.1, 0.1, 0.1])
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.SINGLE_FINGER, \
            f"Thumb should be ignored, got {r.gesture.name}"

    def test_single_finger_mid_semi(self):
        """Middle finger semi (0.50) still ok for SINGLE_FINGER (tendon coupling)"""
        lm = _make_landmarks([0.1, 0.5, 0.25, 0.1, 0.1])
        # mid = 0.50 normalized < single_mid_max(0.60) ✓
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.SINGLE_FINGER, \
            f"Middle semi should be tolerated, got {r.gesture.name}"

    def test_single_finger_mid_too_high(self):
        """Middle above relaxed bound → not SINGLE_FINGER"""
        lm = _make_landmarks([0.1, 0.5, 0.32, 0.1, 0.1])
        # mid = 0.64 normalized > single_mid_max(0.60) → fails
        r = self.engine.classify(lm)
        # Falls through to TWO_FINGER (index+middle both > fist_threshold)
        assert r.gesture == Gesture.TWO_FINGER, \
            f"Expected TWO_FINGER, got {r.gesture.name}"

    def test_single_finger_idx_not_dominant(self):
        """Index and middle both semi-extended, ratio too low → TWO_FINGER.
        Prevents SINGLE_FINGER stealing from a relaxed peace sign."""
        lm = _make_landmarks([0.1, 0.4, 0.32, 0.1, 0.1])
        # index=0.80, middle=0.64, ratio=1.25 < 1.3 → fails dominance
        r = self.engine.classify(lm)
        assert r.gesture == Gesture.TWO_FINGER, \
            f"Expected TWO_FINGER (ratio guard), got {r.gesture.name}"
