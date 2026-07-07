"""Tests for OneEuroFilter and CursorController."""
import numpy as np
import pytest
from microgesture.pipeline.cursor import OneEuroFilter, CursorController


class TestOneEuroFilter:
    def test_first_value_returned(self):
        f = OneEuroFilter()
        assert f.filter(0.5) == 0.5

    def test_smoothing_reduces_noise(self):
        f = OneEuroFilter(beta=0.001, fcmin=1.0, min_cutoff=1.0, fps=30.0)
        values = [0.5] * 10 + [0.5 + np.sin(i * 0.5) * 0.02 for i in range(10)]
        raw_std = np.std(values[-10:])
        smoothed = [f.filter(v) for v in values]
        smoothed_std = np.std(smoothed[-10:])
        assert smoothed_std < raw_std, \
            f"Smoothing should reduce std: raw={raw_std:.4f} > smooth={smoothed_std:.4f}"

    def test_reset(self):
        f = OneEuroFilter()
        f.filter(0.5)
        f.reset()
        assert f._initialized is False


class TestCursorController:
    def setup_method(self):
        self.cursor = CursorController(
            screen_width=1920, screen_height=1080,
            sensitivity=1.0, deadzone=0.0,
        )

    def test_frozen_returns_zero(self):
        self.cursor.update(0.5, 0.5)  # init
        self.cursor.freeze()
        dx, dy = self.cursor.update(0.6, 0.5)
        assert dx == 0.0 and dy == 0.0

    def test_unfreeze_resumes(self):
        self.cursor.update(0.5, 0.5)
        self.cursor.freeze()
        self.cursor.update(0.6, 0.5)
        self.cursor.unfreeze()
        self.cursor.update(0.5, 0.5)  # re-init
        dx, dy = self.cursor.update(0.7, 0.5)
        assert dx != 0.0, "Cursor should move after unfreeze"

    def test_deadzone_suppresses_tiny_moves(self):
        c = CursorController(screen_width=1920, screen_height=1080,
                             sensitivity=1.0, deadzone=0.01)
        c.update(0.5, 0.5)  # init
        # Tiny move: 0.002 * 1920 = 3.84px < deadzone 19.2px
        dx, dy = c.update(0.502, 0.5)
        assert dx == 0.0, f"Tiny move should be deadzone'd, got dx={dx:.1f}"

    def test_large_move_passes_deadzone(self):
        c = CursorController(screen_width=1920, screen_height=1080,
                             sensitivity=1.0, deadzone=0.01)
        # Build up steady movement so the 1€ filter tracks velocity
        for x in [0.5, 0.52, 0.54, 0.56, 0.58, 0.60]:
            dx, _ = c.update(x, 0.5)
        # After filtering settles, a steady move of 0.02 * 1920 = 38.4 > 19.2px
        assert dx != 0.0, f"Steady movement should pass deadzone, got dx={dx:.1f}"

    def test_tap_deadzone_larger(self):
        c = CursorController(screen_width=1920, screen_height=1080,
                             sensitivity=1.0, deadzone=0.0, tap_deadzone=0.05)
        # Let filter settle
        for _ in range(5):
            c.update(0.5, 0.5)

        # Normal mode (no deadzone) → should move
        c._tap_active = False
        dx_norm, _ = c.update(0.52, 0.5)
        assert dx_norm != 0.0, f"Normal should move, got dx={dx_norm:.1f}"

        # Tap mode (large deadzone) → should be suppressed
        for _ in range(5):
            c.update(0.5, 0.5)
        c._tap_active = True
        dx_tap, _ = c.update(0.52, 0.5)
        assert dx_tap == 0.0, f"Tap should suppress, got dx={dx_tap:.1f}"
