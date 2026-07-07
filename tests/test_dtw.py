"""Tests for motion-based DtwMatcher and DtwTrainer."""
import numpy as np
import pytest
from microgesture.recognition.dtw_matcher import (
    DtwMatcher, DtwState, _normalize_wrist, _dtw_distance,
)
from microgesture.recognition.dtw_trainer import DtwTrainer, TrainerState


class TestNormalizeWrist:
    def test_single_frame(self):
        lm = np.zeros((21, 3), dtype=np.float32)
        lm[0] = [0.5, 0.5, 0.1]
        lm[8] = [0.6, 0.4, 0.0]
        normed = _normalize_wrist(lm)
        assert normed.shape == (63,)
        assert normed[0] == pytest.approx(0.0, abs=1e-6)

    def test_batch(self):
        lms = np.random.randn(5, 21, 3).astype(np.float32)
        lms[:, 0, :] = [0.5, 0.5, 0]
        normed = _normalize_wrist(lms)
        assert normed.shape == (5, 63)


class TestDtwDistance:
    def test_same_sequence_zero(self):
        seq = np.random.randn(10, 63).astype(np.float32)
        assert _dtw_distance(seq, seq) < 0.01

    def test_different_lengths(self):
        assert _dtw_distance(np.random.randn(10, 63).astype(np.float32),
                             np.random.randn(15, 63).astype(np.float32)) > 0


class TestDtwMatcherMotion:
    def setup_method(self):
        self.m = DtwMatcher()
        self.m._motion_threshold = 0.005
        self.m._still_frames = 5
        self.m._min_record_frames = 8
        self.m._max_record_frames = 120
        self._tip_x = 0.5

    def make_lm(self, dx=0.0):
        """Create landmarks with controlled tip movement.
        dx is the displacement FROM THE PREVIOUS frame's tip position."""
        self._tip_x += dx
        lm = np.zeros((21, 3), dtype=np.float32)
        lm[8] = [self._tip_x, 0.5, 0]
        lm[0] = [0.5, 0.8, 0]
        return lm

    def test_idle_stays_idle(self):
        self.m.feed(self.make_lm(0))
        self.m.feed(self.make_lm(0))
        assert self.m.state == DtwState.IDLE

    def test_motion_triggers_moving(self):
        self.m.feed(self.make_lm(0))       # init prev_tip
        self.m.feed(self.make_lm(0.01))    # big move
        assert self.m.state == DtwState.MOVING

    def test_small_motion_stays_idle(self):
        self.m.feed(self.make_lm(0))
        self.m.feed(self.make_lm(0.002))   # too small
        assert self.m.state == DtwState.IDLE

    def test_still_after_motion_stops(self):
        self.m._still_frames = 3
        self.m._min_record_frames = 3
        self.m.feed(self.make_lm(0))
        for _ in range(5):
            self.m.feed(self.make_lm(0.01))
        assert self.m.state == DtwState.MOVING
        # Stop: still frames (velocity < threshold)
        for _ in range(5):
            self.m.feed(self.make_lm(0))  # zero velocity
        assert self.m.state == DtwState.IDLE, \
            f"Expected IDLE, got {self.m.state.name}"

    def test_full_gesture_cycle(self):
        self.m._still_frames = 5
        self.m._min_record_frames = 5
        self.m._match_threshold = 999

        self.m.add_template("test", "Test",
                            np.random.randn(10, 63).astype(np.float32))

        self.m.feed(self.make_lm(0))
        for _ in range(20):
            self.m.feed(self.make_lm(0.01))
        for _ in range(5):
            self.m.feed(self.make_lm(0))  # zero velocity
        assert self.m.state == DtwState.IDLE, \
            f"Expected IDLE, got {self.m.state.name}"

    def test_no_match_without_templates(self):
        self.m._still_frames = 3
        self.m._min_record_frames = 5
        self.m.feed(self.make_lm(0))
        for _ in range(10):
            self.m.feed(self.make_lm(0.01))
        match = None
        for _ in range(5):
            r = self.m.feed(self.make_lm(0))
            if r:
                match = r
        assert match is None


class TestDtwTrainer:
    def setup_method(self):
        self.t = DtwTrainer()
        self.t._motion_threshold = 0.005
        self.t._still_frames = 3
        self.t._min_frames = 5
        self.t._max_frames = 120
        self.t.READY_SECONDS = 0  # skip countdown

    def still_lm(self):
        lm = np.zeros((21, 3), dtype=np.float32)
        lm[8] = [0.5, 0.5, 0]
        lm[0] = [0.5, 0.8, 0]
        return lm

    def test_three_takes_dba(self):
        self.t.start("g1", "G1")
        # Use controlled tip movement: start at 0.5, move right, stop
        for take in range(3):
            self.t._state = TrainerState.RECORDING
            self.t._buffer.clear()
            self.t._prev_tip = None
            self.t._still_counter = 0
            self.t._is_moving = True  # skip the "wait for motion" phase
            # Move: 10 frames, tip moves right
            for i in range(10):
                lm = np.zeros((21, 3), dtype=np.float32)
                lm[8] = [0.5 + i * 0.01, 0.5, 0]
                lm[0] = [0.5, 0.8, 0]
                n, _ = self.t.feed(lm)
            # Still: 3 frames, same tip position
            for _ in range(3):
                lm = np.zeros((21, 3), dtype=np.float32)
                lm[8] = [0.5 + 0.09, 0.5, 0]  # same as last moving frame
                lm[0] = [0.5, 0.8, 0]
                n, _ = self.t.feed(lm)
            assert n == take + 1, f"Expected take {take + 1}, got {n}"
        result = self.t.finish()
        assert result is not None
        assert len(result.raw_takes) == 3
        assert result.sequence.shape[1] == 63

    def test_cancel(self):
        self.t.start("t", "T")
        self.t._state = TrainerState.RECORDING
        self.t._is_moving = True
        for i in range(8):
            lm = np.zeros((21, 3), dtype=np.float32)
            lm[8] = [0.5 + i * 0.01, 0.5, 0]
            lm[0] = [0.5, 0.8, 0]
            self.t.feed(lm)
        self.t.cancel()
        assert self.t.finish() is None
