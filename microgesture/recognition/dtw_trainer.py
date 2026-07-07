"""Gesture template trainer: motion-based 3-take recording + DBA averaging.

Same motion-based segmentation as DtwMatcher:
  still → movement → still (automatic)

Guides user through 3 recordings, each with a countdown:
  1. "Ready..." (2s) → "Go!" → perform gesture → stop moving
  2. After 3 takes: DBA average → TrainerResult
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np

from .dtw_matcher import _dtw_distance, _normalize_wrist

logger = logging.getLogger(__name__)


class TrainerState(Enum):
    IDLE = auto()
    READY = auto()      # countdown before recording starts
    RECORDING = auto()  # buffering landmarks during motion
    DONE = auto()       # all 3 takes complete


@dataclass
class TrainerResult:
    sequence: np.ndarray       # (N, 63) DBA-averaged template
    raw_takes: list[np.ndarray]  # 3 raw recordings
    name: str
    label: str


class DtwTrainer:
    """Motion-based guided registration of a custom gesture.

    Lifecycle per take:
      start() → READY (2s countdown) → "Go!"
      User: still → move → still
      Trainer detects motion → buffers → still → saves take
      Repeat 3x → finish() → DBA average
    """

    TAKES_REQUIRED = 3
    READY_SECONDS = 3
    FIRST_READY_SECONDS = 5

    def __init__(self, config=None):
        self._state = TrainerState.IDLE
        self._buffer: deque[np.ndarray] = deque()
        self._takes: list[np.ndarray] = []
        self._name = ""
        self._label = ""
        self._on_take: Optional[Callable[[int], None]] = None
        self._on_countdown: Optional[Callable[[int], None]] = None

        # Motion thresholds (same as DtwMatcher)
        self._motion_threshold = 0.005
        self._still_frames = 10
        self._max_frames = 120
        self._min_frames = 15
        self._dtw_radius = 10
        self._ready_end = 0.0
        self._still_counter = 0
        self._prev_tip: Optional[np.ndarray] = None
        self._is_moving = False

        if config is not None:
            self._motion_threshold = config.get("dtw", "motion_threshold", default=self._motion_threshold)
            self._still_frames = config.get("dtw", "still_frames", default=self._still_frames)
            self._max_frames = config.get("dtw", "max_record_frames", default=self._max_frames)
            self._min_frames = config.get("dtw", "min_record_frames", default=self._min_frames)
            self._dtw_radius = config.get("dtw", "dtw_radius", default=self._dtw_radius)

    # ── public API ────────────────────────────────────────────────────────────

    def start(self, name: str, label: str,
              on_take: Callable[[int], None] | None = None,
              on_countdown: Callable[[int], None] | None = None) -> None:
        self._name = name
        self._label = label
        self._state = TrainerState.READY
        self._ready_end = time.time() + self.FIRST_READY_SECONDS
        self._buffer.clear()
        self._takes.clear()
        self._still_counter = 0
        self._prev_tip = None
        self._is_moving = False
        self._on_take = on_take
        self._on_countdown = on_countdown
        logger.info("Trainer started: %s (take %d/3)", name, len(self._takes) + 1)

    def feed(self, landmarks: np.ndarray) -> tuple[Optional[int], Optional[str]]:
        """Feed one frame. Returns (take_number, status_message).

        status_message: current state description for UI overlay.
        """
        tip = landmarks[8].copy()
        velocity = 0.0
        if self._prev_tip is not None:
            velocity = float(np.linalg.norm(tip - self._prev_tip))
        self._prev_tip = tip
        is_moving = velocity > self._motion_threshold

        if self._state == TrainerState.IDLE:
            return (None, "空闲")
        elif self._state == TrainerState.READY:
            return self._handle_ready()
        elif self._state == TrainerState.RECORDING:
            return self._handle_recording(landmarks, is_moving)
        elif self._state == TrainerState.DONE:
            return (None, f"完成 ({len(self._takes)}/3 次)")
        return (None, "")

    def cancel(self) -> None:
        self._state = TrainerState.IDLE
        self._buffer.clear()
        self._takes.clear()
        logger.info("Trainer cancelled")

    def finish(self) -> Optional[TrainerResult]:
        if len(self._takes) < self.TAKES_REQUIRED:
            return None
        avg_sequence = self._compute_dba()
        self._state = TrainerState.IDLE
        return TrainerResult(
            sequence=avg_sequence,
            raw_takes=list(self._takes),
            name=self._name, label=self._label,
        )

    # ── state machine ─────────────────────────────────────────────────────────

    def _handle_ready(self) -> tuple[Optional[int], Optional[str]]:
        remaining = max(0, self._ready_end - time.time())
        if self._on_countdown:
            self._on_countdown(int(remaining) + 1)
        if remaining <= 0:
            self._state = TrainerState.RECORDING
            self._buffer.clear()
            self._still_counter = 0
            self._is_moving = False
            self._prev_tip = None
            return (None, "开始! 做手势...")
        return (None, f"准备... {int(remaining) + 1}")

    def _handle_recording(self, landmarks, is_moving) -> tuple[Optional[int], Optional[str]]:
        # Timeout
        if len(self._buffer) >= self._max_frames:
            self._buffer.clear()
            self._state = TrainerState.IDLE
            return (None, "超时 — 请重试")

        if not self._is_moving and is_moving:
            self._is_moving = True

        if self._is_moving:
            self._buffer.append(_normalize_wrist(landmarks))

        # Detect end: stopped moving for still_frames
        if self._is_moving and not is_moving:
            self._still_counter += 1
            if self._still_counter >= self._still_frames:
                if len(self._buffer) >= self._min_frames:
                    return self._save_take()
                else:
                    self._buffer.clear()
                    self._state = TrainerState.IDLE
                    return (None, f"太短 ({len(self._buffer)} 帧) — 请重试")
        else:
            self._still_counter = 0

        buf_len = len(self._buffer)
        if self._is_moving:
            return (None, f"录制中... {buf_len} 帧")
        else:
            return (None, "等待动作...")

    def _save_take(self) -> tuple[Optional[int], Optional[str]]:
        seq = np.stack(list(self._buffer))
        self._takes.append(seq)
        take_num = len(self._takes)
        self._buffer.clear()
        self._still_counter = 0
        self._is_moving = False
        self._prev_tip = None

        if self._on_take:
            self._on_take(take_num)

        if take_num >= self.TAKES_REQUIRED:
            self._state = TrainerState.DONE
            return (take_num, f"第{take_num}/3次 — 完成!")
        else:
            self._state = TrainerState.READY
            self._ready_end = time.time() + self.READY_SECONDS  # shorter for 2nd/3rd
            return (take_num, f"第{take_num}/3次 — 准备...")

    # ── DBA ───────────────────────────────────────────────────────────────────

    def _compute_dba(self, iterations: int = 5) -> np.ndarray:
        if len(self._takes) == 1:
            return self._takes[0].astype(np.float32)
        lengths = [len(t) for t in self._takes]
        median_idx = np.argsort(lengths)[len(lengths) // 2]
        barycenter = self._takes[median_idx].astype(np.float64).copy()

        for it in range(iterations):
            acc: list[list[np.ndarray]] = [[] for _ in range(len(barycenter))]
            for take in self._takes:
                _, path = _get_dtw_path(take, barycenter, radius=self._dtw_radius)
                for i, j in path:
                    if j < len(acc):
                        acc[j].append(take[i])
            new_bary = np.zeros_like(barycenter)
            for j, frames in enumerate(acc):
                new_bary[j] = np.mean(frames, axis=0) if frames else barycenter[j]
            barycenter = new_bary
        return barycenter.astype(np.float32)


def _get_dtw_path(seq1, seq2, radius=10):
    try:
        from fastdtw import fastdtw
        return fastdtw(seq1, seq2, radius=radius)
    except ImportError:
        if not getattr(_get_dtw_path, "_warned", False):
            logger.warning("fastdtw not available, DBA alignment degraded (install: pip install fastdtw)")
            _get_dtw_path._warned = True
        n1, n2 = len(seq1), len(seq2)
        path = [(min(i, n1 - 1), min(i, n2 - 1)) for i in range(max(n1, n2))]
        return _dtw_distance(seq1, seq2, radius=radius), path
