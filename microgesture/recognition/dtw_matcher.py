"""DTW custom gesture matcher — motion-based segmentation.

Protocol: still → movement → still (automatic, no fist required).
The state machine detects motion onset/offset to segment gesture sequences.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── data types ────────────────────────────────────────────────────────────────


class DtwState(Enum):
    IDLE = auto()     # waiting for motion
    MOVING = auto()   # gesture in progress, buffering landmarks
    STILL = auto()    # motion stopped, confirming end-of-gesture


@dataclass
class Match:
    """Result of a successful DTW match."""
    name: str
    label: str
    distance: float
    confidence: float
    sequence_length: int
    action_config: dict = field(default_factory=dict)


@dataclass
class Template:
    """A stored gesture template for matching."""
    name: str
    label: str
    sequence: np.ndarray    # (N, 63) wrist-relative flattened landmarks
    action_config: dict = field(default_factory=dict)


# ── DTW distance ──────────────────────────────────────────────────────────────


def _dtw_distance(seq1: np.ndarray, seq2: np.ndarray, radius: int = 10) -> float:
    """Path-length-normalized DTW distance between two landmark sequences."""
    try:
        from fastdtw import fastdtw
        distance, path = fastdtw(seq1, seq2, radius=radius)
        return distance / max(len(path), 1)
    except ImportError:
        n1, n2 = len(seq1), len(seq2)
        if n1 < n2:
            seq1 = np.concatenate([seq1, np.tile(seq1[-1:], (n2 - n1, 1))])
        elif n2 < n1:
            seq2 = np.concatenate([seq2, np.tile(seq2[-1:], (n1 - n2, 1))])
        return float(np.linalg.norm(seq1 - seq2) / max(n1, n2))


def _normalize_wrist(landmarks: np.ndarray) -> np.ndarray:
    """Subtract wrist position for translation invariance. Returns 63-dim."""
    if landmarks.ndim == 2:
        return (landmarks - landmarks[0]).flatten()
    return (landmarks - landmarks[:, 0:1, :]).reshape(landmarks.shape[0], -1)


# ── Matcher ───────────────────────────────────────────────────────────────────


class DtwMatcher:
    """Gesture sequence matcher using motion-based segmentation + DTW.

    Lifecycle per gesture:
      IDLE → (fingertip velocity > threshold) → MOVING → buffering
      MOVING → (velocity < threshold for N consecutive frames) → STILL
      STILL → N consecutive still frames → compute DTW → IDLE

    No fist protocol needed — the user simply performs the gesture and pauses.
    """

    def __init__(self, config=None, templates_path: str | Path | None = None):
        # ── motion detection ────────────────────────────────────────────
        self._motion_threshold = 0.005
        self._still_frames = 10
        self._min_record_frames = 15
        self._max_record_frames = 120

        # ── matching ────────────────────────────────────────────────────
        self._dtw_radius = 10
        self._match_threshold = 8.0
        self._cooldown_frames = 90

        if config is not None:
            self._motion_threshold = config.get("dtw", "motion_threshold", default=self._motion_threshold)
            self._still_frames = config.get("dtw", "still_frames", default=self._still_frames)
            self._min_record_frames = config.get("dtw", "min_record_frames", default=self._min_record_frames)
            self._max_record_frames = config.get("dtw", "max_record_frames", default=self._max_record_frames)
            self._dtw_radius = config.get("dtw", "dtw_radius", default=self._dtw_radius)
            self._match_threshold = config.get("dtw", "match_threshold", default=self._match_threshold)
            self._cooldown_frames = config.get("dtw", "cooldown_frames", default=self._cooldown_frames)

        # ── state ───────────────────────────────────────────────────────
        self._state = DtwState.IDLE
        self._buffer: deque[np.ndarray] = deque()
        self._still_counter = 0
        self._cooldown = 0
        self._prev_tip: Optional[np.ndarray] = None
        self._templates: list[Template] = []

        # ── templates path ──────────────────────────────────────────────
        if templates_path is not None:
            self._templates_path = Path(templates_path)
        elif config is not None:
            path_str = config.get("dtw", "templates_path", default="templates.json")
            self._templates_path = Path(__file__).parent.parent / "training" / path_str
        else:
            self._templates_path = Path(__file__).parent.parent / "training" / "templates.json"

        self._load_templates()

    # ── public API ────────────────────────────────────────────────────────────

    def feed(self, landmarks: np.ndarray) -> Optional[Match]:
        """Feed one frame. Returns Match when a gesture is recognised, else None.

        Args:
            landmarks: (21, 3) array from MediaPipe.
        """
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        # ── fingertip velocity ──────────────────────────────────────────
        tip = landmarks[8].copy()  # index fingertip
        velocity = 0.0
        if self._prev_tip is not None:
            velocity = float(np.linalg.norm(tip - self._prev_tip))
        self._prev_tip = tip

        is_moving = velocity > self._motion_threshold

        # ── state machine ───────────────────────────────────────────────
        if self._state == DtwState.IDLE:
            return self._handle_idle(landmarks, is_moving)
        elif self._state == DtwState.MOVING:
            return self._handle_moving(landmarks, is_moving)
        elif self._state == DtwState.STILL:
            return self._handle_still(landmarks, is_moving)

        return None

    @property
    def state(self) -> DtwState:
        return self._state

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def template_count(self) -> int:
        return len(self._templates)

    # ── state machine ─────────────────────────────────────────────────────────

    def _handle_idle(self, landmarks: np.ndarray, is_moving: bool) -> Optional[Match]:
        if is_moving:
            self._state = DtwState.MOVING
            self._buffer.clear()
            self._buffer.append(_normalize_wrist(landmarks))
        return None

    def _handle_moving(self, landmarks: np.ndarray, is_moving: bool) -> Optional[Match]:
        # Buffer the landmark
        self._buffer.append(_normalize_wrist(landmarks))

        # Timeout check
        if len(self._buffer) >= self._max_record_frames:
            logger.debug("DTW: MOVING → IDLE (timeout %d frames)", len(self._buffer))
            self._buffer.clear()
            self._state = DtwState.IDLE
            return None

        if not is_moving:
            self._still_counter = 1
            self._state = DtwState.STILL
            logger.debug("DTW: MOVING → STILL (motion stopped)")

        return None

    def _handle_still(self, landmarks: np.ndarray, is_moving: bool) -> Optional[Match]:
        if is_moving:
            # Resume recording — the pause was brief
            self._buffer.append(_normalize_wrist(landmarks))
            self._state = DtwState.MOVING
            self._still_counter = 0
            return None

        self._still_counter += 1

        if self._still_counter >= self._still_frames:
            # Confirmed end of gesture → attempt match
            if len(self._buffer) < self._min_record_frames:
                logger.debug("DTW: sequence too short (%d < %d)", len(self._buffer), self._min_record_frames)
                self._buffer.clear()
                self._state = DtwState.IDLE
                return None

            logger.debug("DTW: matching %d frames against %d templates",
                         len(self._buffer), len(self._templates))
            match = self._compute_match()
            self._buffer.clear()
            self._state = DtwState.IDLE
            if match:
                self._cooldown = self._cooldown_frames
            return match

        return None

    # ── matching ──────────────────────────────────────────────────────────────

    def _compute_match(self) -> Optional[Match]:
        sequence = np.stack(list(self._buffer))  # (T, 63)
        best_dist = float("inf")
        best_template: Optional[Template] = None

        for tmpl in self._templates:
            dist = _dtw_distance(sequence, tmpl.sequence, radius=self._dtw_radius)
            logger.debug("DTW: distance to '%s' = %.2f", tmpl.name, dist)
            if dist < best_dist:
                best_dist = dist
                best_template = tmpl

        if best_template is None or best_dist >= self._match_threshold:
            logger.debug("DTW: no match (best=%.2f >= %.2f)", best_dist, self._match_threshold)
            return None

        confidence = max(0.0, 1.0 - best_dist / self._match_threshold)
        logger.info("DTW match: %s (dist=%.2f, conf=%.2f)", best_template.name, best_dist, confidence)
        return Match(
            name=best_template.name,
            label=best_template.label,
            distance=best_dist,
            confidence=confidence,
            sequence_length=len(sequence),
            action_config=best_template.action_config,
        )

    # ── template management ───────────────────────────────────────────────────

    def _load_templates(self) -> None:
        if not self._templates_path.exists():
            self._templates = []
            return
        try:
            with open(self._templates_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._templates = [
                Template(
                    name=t["name"], label=t["label"],
                    sequence=np.array(t["sequence"], dtype=np.float32),
                    action_config=t.get("action", {}),
                )
                for t in data.get("templates", [])
            ]
            logger.info("Loaded %d DTW templates", len(self._templates))
        except Exception:
            logger.exception("Failed to load templates")
            self._templates = []

    def add_template(self, name: str, label: str, sequence: np.ndarray,
                     action_config: dict | None = None) -> None:
        if action_config is None:
            action_config = {}
        self._templates.append(Template(
            name=name, label=label,
            sequence=sequence.astype(np.float32),
            action_config=action_config,
        ))
        self._save_templates()
        logger.info("Template added: %s (%d frames)", name, len(sequence))

    def remove_template(self, name: str) -> None:
        self._templates = [t for t in self._templates if t.name != name]
        self._save_templates()
        logger.info("Template removed: %s", name)

    def _save_templates(self) -> None:
        self._templates_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "templates": [
                {"name": t.name, "label": t.label,
                 "sequence": t.sequence.tolist(), "action": t.action_config}
                for t in self._templates
            ],
        }
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self._templates_path.parent), prefix=".templates_", suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, str(self._templates_path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def close(self) -> None:
        pass
