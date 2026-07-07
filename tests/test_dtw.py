"""Tests for DtwMatcher state machine and DtwTrainer DBA."""
import numpy as np
import pytest
from microgesture.recognition.dtw_matcher import (
    DtwMatcher, DtwState, _normalize_wrist, _dtw_distance,
)
from microgesture.recognition.dtw_trainer import DtwTrainer


class MockGesture:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name


FIST = MockGesture("FIST")
PALM = MockGesture("PALM_OPEN")


class TestNormalizeWrist:
    def test_single_frame(self):
        lm = np.zeros((21, 3), dtype=np.float32)
        lm[0] = [0.5, 0.5, 0.1]
        lm[8] = [0.6, 0.4, 0.0]
        normed = _normalize_wrist(lm)
        assert normed.shape == (63,)
        assert normed[0] == 0.0  # wrist x = 0
        assert normed[1] == 0.0  # wrist y = 0
        assert normed[2] == 0.0  # wrist z = 0
        assert normed[24] == pytest.approx(0.1, abs=1e-6)  # index tip x = 0.6 - 0.5

    def test_batch(self):
        lms = np.random.randn(5, 21, 3).astype(np.float32)
        lms[:, 0, :] = [0.5, 0.5, 0]  # wrist position
        normed = _normalize_wrist(lms)
        assert normed.shape == (5, 63)


class TestDtwDistance:
    def test_same_sequence_zero_distance(self):
        seq = np.random.randn(10, 63).astype(np.float32)
        d = _dtw_distance(seq, seq)
        assert d < 0.01, f"Same sequence should have ~0 DTW distance, got {d:.4f}"

    def test_different_lengths(self):
        seq1 = np.random.randn(10, 63).astype(np.float32)
        seq2 = np.random.randn(15, 63).astype(np.float32)
        d = _dtw_distance(seq1, seq2)
        assert d > 0, "Different sequences should have non-zero distance"


class TestDtwMatcherStateMachine:
    def setup_method(self):
        self.m = DtwMatcher()
        self.m._arm_frames = 3
        self.m._min_record_frames = 5

    def dummy_landmarks(self):
        return np.zeros((21, 3), dtype=np.float32)

    def test_idle_stays_idle(self):
        r = self.m.feed(self.dummy_landmarks(), PALM)
        assert r is None
        assert self.m.state == DtwState.IDLE

    def test_fist_enters_arming(self):
        self.m.feed(self.dummy_landmarks(), FIST)
        assert self.m.state == DtwState.ARMING

    def test_early_release_aborts(self):
        self.m.feed(self.dummy_landmarks(), FIST)
        self.m.feed(self.dummy_landmarks(), PALM)
        assert self.m.state == DtwState.IDLE, \
            f"Early FIST release should abort, got {self.m.state.name}"

    def test_hold_fist_arms(self):
        for _ in range(3):
            self.m.feed(self.dummy_landmarks(), FIST)
        assert self.m.state == DtwState.RECORDING

    def test_recording_too_short_no_match(self):
        # Arm
        for _ in range(3):
            self.m.feed(self.dummy_landmarks(), FIST)
        # Record 2 frames (< min_record_frames=5)
        for _ in range(2):
            self.m.feed(self.dummy_landmarks(), PALM)
        # Close fist
        r = self.m.feed(self.dummy_landmarks(), FIST)
        assert r is None, "Too-short sequence should not match"
        assert self.m.state == DtwState.IDLE

    def test_match_with_similar_template(self):
        m = DtwMatcher()
        m._arm_frames = 2
        m._min_record_frames = 3
        m._match_threshold = 999

        # Register a template
        template = np.random.randn(10, 63).astype(np.float32)
        m.add_template("test", "Test", template)

        # Arm
        for _ in range(2):
            m.feed(np.zeros((21, 3), dtype=np.float32), FIST)

        # Record similar sequence
        noisy = template + np.random.normal(0, 0.001, template.shape).astype(np.float32)
        for i in range(len(noisy)):
            # Reconstruct landmarks from normalized features for feed()
            lm = np.zeros((21, 3), dtype=np.float32)
            lm[:, 0] = noisy[i, 0:21]  # x coords
            lm[:, 1] = noisy[i, 21:42]  # y coords
            lm[:, 2] = noisy[i, 42:63]  # z coords
            r = m.feed(lm, PALM)

        # Close fist
        r = m.feed(np.zeros((21, 3), dtype=np.float32), FIST)
        assert r is not None, "Should match similar template"
        assert r.name == "test"

    def test_no_match_without_templates(self):
        m = DtwMatcher()
        m._arm_frames = 2
        m._min_record_frames = 3
        for _ in range(2):
            m.feed(np.zeros((21, 3), dtype=np.float32), FIST)
        for _ in range(5):
            m.feed(np.random.randn(21, 3).astype(np.float32), PALM)
        r = m.feed(np.zeros((21, 3), dtype=np.float32), FIST)
        assert r is None, "No templates → no match"


class TestDtwTrainer:
    def setup_method(self):
        self.t = DtwTrainer()
        self.t._arm_frames = 2
        self.t._min_frames = 3

    def dummy_landmarks(self):
        return np.zeros((21, 3), dtype=np.float32)

    def test_start_resets_state(self):
        self.t.start("test", "Test")
        assert self.t._state.name == "IDLE"

    def test_three_takes_records(self):
        self.t.start("g1", "G1")
        for take in range(3):
            # Arm
            for _ in range(2):
                n = self.t.feed(self.dummy_landmarks(), FIST)
            assert n is None
            # Record gesture (random frames)
            for _ in range(5):
                lm = np.random.randn(21, 3).astype(np.float32)
                n = self.t.feed(lm, PALM)
            # End
            n = self.t.feed(self.dummy_landmarks(), FIST)
            assert n == take + 1, f"Expected take {take + 1}, got {n}"

        result = self.t.finish()
        assert result is not None
        assert result.name == "g1"
        assert len(result.raw_takes) == 3
        assert result.sequence.shape[1] == 63
        # DBA average should be between min and max take lengths
        take_lens = [len(t) for t in result.raw_takes]
        assert min(take_lens) <= len(result.sequence) <= max(take_lens), \
            f"DBA length {len(result.sequence)} not in range [{min(take_lens)}, {max(take_lens)}]"

    def test_cancel_discards(self):
        self.t.start("test", "Test")
        for _ in range(2):
            self.t.feed(self.dummy_landmarks(), FIST)
        for _ in range(5):
            self.t.feed(np.random.randn(21, 3).astype(np.float32), PALM)
        self.t.feed(self.dummy_landmarks(), FIST)  # take 1 recorded
        self.t.cancel()
        r = self.t.finish()
        assert r is None, "Cancelled trainer should return None from finish()"
