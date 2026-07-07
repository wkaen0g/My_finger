"""DTW-based custom gesture matcher with FIST-delimited segmentation.

Protocol: FIST (arm) → gesture motion (record) → FIST (end).
The state machine segments gesture sequences and matches them against
stored templates using fastdtw.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── data types ────────────────────────────────────────────────────────────────


class DtwState(Enum):
    IDLE = auto()
    ARMING = auto()
    RECORDING = auto()


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
    sequence: np.ndarray   # (N, 63) wrist-relative flattened landmarks
    action_config: dict = field(default_factory=dict)


# ── DTW distance ──────────────────────────────────────────────────────────────


def _dtw_distance(seq1: np.ndarray, seq2: np.ndarray, radius: int = 10) -> float:
    """Normalized DTW distance between two landmark sequences.

    Args:
        seq1, seq2: Arrays of shape (N, 63) or (N, D).
        radius: fastdtw sakoe-chiba band radius.

    Returns:
        Path-length-normalized DTW distance.
    """
    try:
        from fastdtw import fastdtw
        distance, path = fastdtw(seq1, seq2, radius=radius)
        return distance / max(len(path), 1)
    except ImportError:
        # Fallback: no fastdtw → use Euclidean
        logger.warning("fastdtw not available, using Euclidean distance")
        # Pad shorter sequence
        n1, n2 = len(seq1), len(seq2)
        if n1 < n2:
            pad = np.zeros((n2 - n1, seq1.shape[1]), dtype=seq1.dtype)
            seq1 = np.concatenate([seq1, pad])
        elif n2 < n1:
            pad = np.zeros((n1 - n2, seq2.shape[1]), dtype=seq2.dtype)
            seq2 = np.concatenate([seq2, pad])
        return float(np.linalg.norm(seq1 - seq2) / max(n1, n2))


def _normalize_wrist(landmarks: np.ndarray) -> np.ndarray:
    """Subtract wrist position for translation invariance.

    Args:
        landmarks: (21, 3) or (N, 21, 3).

    Returns:
        Flattened 63-dim or (N, 63) with wrist subtracted.
    """
    if landmarks.ndim == 2:
        return (landmarks - landmarks[0]).flatten()
    return (landmarks - landmarks[:, 0:1, :]).reshape(landmarks.shape[0], -1)


# ── Matcher ───────────────────────────────────────────────────────────────────


class DtwMatcher:
    """Gesture sequence matcher using FIST-delimited segmentation + DTW.

    Lifecycle per gesture:
      IDLE → (FIST for arm_frames) → ARMING → (FIST released) → RECORDING
      RECORDING → (FIST again or timeout) → compute DTW match → IDLE

    Attributes:
        state: Current DtwState (exposed for UI overlay).
        template_count: Number of loaded templates.
        is_armed: True when state is ARMING or RECORDING.
    """

    def __init__(self, config=None, templates_path: str | Path | None = None):
        """Initialise the matcher.

        Args:
            config: Optional Config-like object with get(*keys, default=...).
            templates_path: Override path to templates.json.
        """
        # ── thresholds ──────────────────────────────────────────────────
        self._arm_frames = 10
        self._max_record_frames = 120
        self._min_record_frames = 15
        self._dtw_radius = 10
        self._match_threshold = 8.0
        self._cooldown_frames = 90

        if config is not None:
            self._arm_frames = config.get("dtw", "arm_frames", default=self._arm_frames)
            self._max_record_frames = config.get("dtw", "max_record_frames", default=self._max_record_frames)
            self._min_record_frames = config.get("dtw", "min_record_frames", default=self._min_record_frames)
            self._dtw_radius = config.get("dtw", "dtw_radius", default=self._dtw_radius)
            self._match_threshold = config.get("dtw", "match_threshold", default=self._match_threshold)
            self._cooldown_frames = config.get("dtw", "cooldown_frames", default=self._cooldown_frames)

        # ── state ───────────────────────────────────────────────────────
        self._state = DtwState.IDLE
        self._buffer: deque[np.ndarray] = deque()
        self._arm_counter = 0
        self._cooldown = 0
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

    def feed(self, landmarks: np.ndarray, gesture) -> Optional[Match]:
        """Feed one frame of hand landmarks + gesture classification.

        Args:
            landmarks: (21, 3) array from MediaPipe.
            gesture: Gesture enum value (used for is_fist detection).

        Returns:
            Match if a gesture was recognised this frame, None otherwise.
        """
        # Cooldown after a match
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        is_fist = (gesture.name == "FIST")

        if self._state == DtwState.IDLE:
            return self._handle_idle(is_fist)
        elif self._state == DtwState.ARMING:
            return self._handle_arming(landmarks, is_fist)
        elif self._state == DtwState.RECORDING:
            return self._handle_recording(landmarks, is_fist)

        return None

    @property
    def state(self) -> DtwState:
        return self._state

    @property
    def is_armed(self) -> bool:
        return self._state in (DtwState.ARMING, DtwState.RECORDING)

    @property
    def template_count(self) -> int:
        return len(self._templates)

    # ── state machine ─────────────────────────────────────────────────────────

    def _handle_idle(self, is_fist: bool) -> Optional[Match]:
        if is_fist:
            self._arm_counter = 1
            self._state = DtwState.ARMING
            logger.debug("DTW: IDLE → ARMING (fist detected)")
        return None

    def _handle_arming(self, landmarks: np.ndarray, is_fist: bool) -> Optional[Match]:
        if is_fist:
            self._arm_counter += 1
            if self._arm_counter >= self._arm_frames:
                self._state = DtwState.RECORDING
                self._buffer.clear()
                logger.debug("DTW: ARMING → RECORDING (armed after %d frames)", self._arm_counter)
        else:
            # Fist released before arming complete → abort
            self._state = DtwState.IDLE
            logger.debug("DTW: ARMING → IDLE (fist released too early, %d/%d frames)",
                         self._arm_counter, self._arm_frames)
        return None

    def _handle_recording(self, landmarks: np.ndarray, is_fist: bool) -> Optional[Match]:
        # Check timeout
        if len(self._buffer) >= self._max_record_frames:
            logger.debug("DTW: RECORDING → IDLE (timeout, %d frames buffered)", len(self._buffer))
            self._buffer.clear()
            self._state = DtwState.IDLE
            return None

        if is_fist:
            # Closing fist → end of gesture
            if len(self._buffer) < self._min_record_frames:
                logger.debug("DTW: sequence too short (%d < %d min)", len(self._buffer), self._min_record_frames)
                self._buffer.clear()
                self._state = DtwState.IDLE
                return None

            logger.debug("DTW: RECORDING → matching (%d frames)", len(self._buffer))
            match = self._compute_match()
            self._buffer.clear()
            self._state = DtwState.IDLE
            if match:
                self._cooldown = self._cooldown_frames
            return match
        else:
            # Buffer the landmark (wrist-normalised, flattened)
            self._buffer.append(_normalize_wrist(landmarks))
            return None

    # ── matching ──────────────────────────────────────────────────────────────

    def _compute_match(self) -> Optional[Match]:
        """Compute DTW distance against all templates, return best match or None."""
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
            logger.debug("DTW: no match (best dist=%.2f >= threshold=%.2f)",
                         best_dist, self._match_threshold)
            return None

        confidence = max(0.0, 1.0 - best_dist / self._match_threshold)
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
        """Load templates from templates.json."""
        if not self._templates_path.exists():
            logger.info("No templates file at %s — starting empty", self._templates_path)
            self._templates = []
            return

        try:
            with open(self._templates_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._templates = [
                Template(
                    name=t["name"],
                    label=t["label"],
                    sequence=np.array(t["sequence"], dtype=np.float32),
                    action_config=t.get("action", {}),
                )
                for t in data.get("templates", [])
            ]
            logger.info("Loaded %d DTW templates from %s", len(self._templates), self._templates_path)
        except Exception:
            logger.exception("Failed to load templates from %s", self._templates_path)
            self._templates = []

    def add_template(self, name: str, label: str, sequence: np.ndarray,
                     action_config: dict | None = None) -> None:
        """Add a template in-memory and persist to disk."""
        if action_config is None:
            action_config = {}
        tmpl = Template(name=name, label=label,
                        sequence=sequence.astype(np.float32),
                        action_config=action_config)
        self._templates.append(tmpl)
        self._save_templates()
        logger.info("Template added: %s (%d frames)", name, len(sequence))

    def remove_template(self, name: str) -> None:
        """Remove a template by name."""
        self._templates = [t for t in self._templates if t.name != name]
        self._save_templates()

    def _save_templates(self) -> None:
        """Write templates to templates.json atomically."""
        import os
        import tempfile

        self._templates_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "templates": [
                {
                    "name": t.name,
                    "label": t.label,
                    "sequence": t.sequence.tolist(),
                    "action": t.action_config,
                }
                for t in self._templates
            ],
        }
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self._templates_path.parent),
            prefix=".templates_", suffix=".tmp",
        )
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

    # ── cleanup ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release resources (no-op for now)."""
        pass
