"""Gesture template trainer: 3-take recording + DBA averaging.

Guides the user through registering a custom gesture:
  1. FIST (arm) → gesture motion → FIST (end) — repeated 3 times
  2. DBA (DTW Barycenter Averaging) computes an average template
  3. Returns TrainerResult for persistence
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np

from .dtw_matcher import _dtw_distance, _normalize_wrist

logger = logging.getLogger(__name__)


class TrainerState(Enum):
    IDLE = auto()
    ARMING = auto()
    RECORDING = auto()


@dataclass
class TrainerResult:
    """Output of a completed training session."""
    sequence: np.ndarray       # (N, 63) DBA-averaged template
    raw_takes: list[np.ndarray]  # 3 raw recordings
    name: str
    label: str


class DtwTrainer:
    """Guided registration of a single custom gesture.

    Lifecycle:
      start(name, label, action_config)   →  begin session
      feed(landmarks, gesture)            →  called per frame, returns take# or None
      cancel()                            →  abort session
      finish()                            →  return TrainerResult

    The caller observes the returned take index to update UI feedback.
    """

    TAKES_REQUIRED = 3

    def __init__(self, config=None):
        self._state = TrainerState.IDLE
        self._buffer: deque[np.ndarray] = deque()
        self._takes: list[np.ndarray] = []
        self._arm_counter = 0
        self._name = ""
        self._label = ""
        self._on_take: Optional[Callable[[int], None]] = None

        # Thresholds
        self._arm_frames = 10
        self._max_frames = 120
        self._min_frames = 15
        self._dtw_radius = 10

        if config is not None:
            self._arm_frames = config.get("dtw", "arm_frames", default=self._arm_frames)
            self._max_frames = config.get("dtw", "max_record_frames", default=self._max_frames)
            self._min_frames = config.get("dtw", "min_record_frames", default=self._min_frames)
            self._dtw_radius = config.get("dtw", "dtw_radius", default=self._dtw_radius)

    # ── public API ────────────────────────────────────────────────────────────

    def start(self, name: str, label: str,
              on_take: Callable[[int], None] | None = None) -> None:
        """Begin a training session.

        Args:
            name: Internal name for the template.
            label: Human-readable display label.
            on_take: Optional callback invoked after each recorded take.
        """
        self._name = name
        self._label = label
        self._state = TrainerState.IDLE
        self._buffer.clear()
        self._takes.clear()
        self._arm_counter = 0
        self._on_take = on_take
        logger.info("Trainer started: name=%s label=%s", name, label)

    def feed(self, landmarks: np.ndarray, gesture) -> Optional[int]:
        """Feed one frame. Returns take number (1-3) when a take is recorded, else None."""
        if self._state == TrainerState.IDLE:
            return self._handle_idle(gesture)
        elif self._state == TrainerState.ARMING:
            return self._handle_arming(landmarks, gesture)
        elif self._state == TrainerState.RECORDING:
            return self._handle_recording(landmarks, gesture)
        return None

    def cancel(self) -> None:
        """Abort current session, discard all takes."""
        self._state = TrainerState.IDLE
        self._buffer.clear()
        self._takes.clear()
        self._arm_counter = 0
        logger.info("Trainer cancelled")

    def finish(self) -> Optional[TrainerResult]:
        """Complete the session, compute DBA average, return result."""
        if len(self._takes) < self.TAKES_REQUIRED:
            logger.warning("Trainer.finish() called with only %d/%d takes",
                           len(self._takes), self.TAKES_REQUIRED)
            return None

        avg_sequence = self._compute_dba()
        self._state = TrainerState.IDLE
        logger.info("Trainer finished: %s (%d frames average)",
                    self._name, len(avg_sequence))
        return TrainerResult(
            sequence=avg_sequence,
            raw_takes=list(self._takes),
            name=self._name,
            label=self._label,
        )

    # ── state machine ─────────────────────────────────────────────────────────

    def _handle_idle(self, gesture) -> Optional[int]:
        if gesture.name == "FIST":
            self._arm_counter = 1
            self._state = TrainerState.ARMING
            logger.debug("Trainer: IDLE → ARMING")
        return None

    def _handle_arming(self, landmarks: np.ndarray, gesture) -> Optional[int]:
        if gesture.name == "FIST":
            self._arm_counter += 1
            if self._arm_counter >= self._arm_frames:
                self._state = TrainerState.RECORDING
                self._buffer.clear()
                logger.debug("Trainer: ARMING → RECORDING (take %d/3)",
                             len(self._takes) + 1)
        else:
            self._state = TrainerState.IDLE
            logger.debug("Trainer: ARMING → IDLE (too early)")
        return None

    def _handle_recording(self, landmarks: np.ndarray, gesture) -> Optional[int]:
        # Timeout check
        if len(self._buffer) >= self._max_frames:
            logger.debug("Trainer: RECORDING → IDLE (timeout)")
            self._buffer.clear()
            self._state = TrainerState.IDLE
            return None

        if gesture.name == "FIST":
            # Closing fist → end of this take
            if len(self._buffer) < self._min_frames:
                logger.debug("Trainer: take too short (%d < %d)", len(self._buffer), self._min_frames)
                self._buffer.clear()
                self._state = TrainerState.IDLE
                return None

            # Save the take
            seq = np.stack(list(self._buffer))
            self._takes.append(seq)
            take_num = len(self._takes)
            self._buffer.clear()
            self._state = TrainerState.IDLE
            logger.info("Trainer: take %d/3 recorded (%d frames)", take_num, len(seq))

            if self._on_take:
                self._on_take(take_num)

            return take_num
        else:
            # Buffer normalized landmarks
            self._buffer.append(_normalize_wrist(landmarks))
            return None

    # ── DBA (DTW Barycenter Averaging) ────────────────────────────────────────

    def _compute_dba(self, iterations: int = 5) -> np.ndarray:
        """Compute DBA average of all recorded takes.

        Petitjean et al. (2011) algorithm:
        1. Pick the median-length take as initial barycenter.
        2. Iterate: DTW-align each take, average aligned frames.

        Args:
            iterations: Number of refinement iterations.

        Returns:
            (N, 63) averaged sequence.
        """
        if len(self._takes) == 1:
            return self._takes[0].astype(np.float32)

        # Initial barycenter = median-length take
        lengths = [len(t) for t in self._takes]
        median_idx = np.argsort(lengths)[len(lengths) // 2]
        barycenter = self._takes[median_idx].astype(np.float64).copy()

        for it in range(iterations):
            # Accumulators: list of lists (one list per barycenter frame)
            acc: list[list[np.ndarray]] = [[] for _ in range(len(barycenter))]

            for take in self._takes:
                _, path = _get_dtw_path(take, barycenter, radius=self._dtw_radius)
                # path: list of (i, j) pairs, i in take, j in barycenter
                for i, j in path:
                    if j < len(acc):
                        acc[j].append(take[i])

            # Average aligned frames
            new_bary = np.zeros_like(barycenter)
            for j, frames in enumerate(acc):
                if frames:
                    new_bary[j] = np.mean(frames, axis=0)
                else:
                    new_bary[j] = barycenter[j]  # keep old if no alignment

            barycenter = new_bary
            logger.debug("DBA iteration %d/%d complete", it + 1, iterations)

        return barycenter.astype(np.float32)


def _get_dtw_path(seq1: np.ndarray, seq2: np.ndarray, radius: int = 10):
    """Get DTW distance and alignment path between two sequences.

    Returns:
        (distance, path) where path is a list of (i, j) index pairs.
    """
    try:
        from fastdtw import fastdtw
        return fastdtw(seq1, seq2, radius=radius)
    except ImportError:
        # Fallback: linear alignment
        n1, n2 = len(seq1), len(seq2)
        path = [(min(i, n1 - 1), min(i, n2 - 1)) for i in range(max(n1, n2))]
        dist = _dtw_distance(seq1, seq2, radius=radius)
        return dist, path
