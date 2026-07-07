"""Shared constants and utilities for HaGRID data loading and classifier training.

This is the SINGLE SOURCE OF TRUTH for:
  - CLASS_MAP: HaGRID raw class name → canonical gesture label
  - GESTURE_LABELS: ordered tuple used for classifier label encoding (order matters!)
  - Feature save logic shared by both HaGRID loaders
  - Package-relative path resolution (independent of CWD)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Canonical label definitions ────────────────────────────────────────────
# HaGRID uses these directory / class names.
# Our system uses the right-hand side as canonical labels.

CLASS_MAP: Dict[str, str] = {
    "palm": "PALM_OPEN",
    "fist": "FIST",
    "peace": "TWO_FINGER",
    "thumb_index": "PINCH",
    "no_gesture": "NO_HAND",
}

# Ordered tuple for classifier label encoding.
# Index 0 → FIST, 1 → NO_HAND, 2 → PALM_OPEN, 3 → PINCH, 4 → TWO_FINGER.
# This order MUST match the Softmax output order of the MLP / ONNX model.
GESTURE_LABELS: Tuple[str, ...] = (
    "FIST",
    "NO_HAND",
    "PALM_OPEN",
    "PINCH",
    "TWO_FINGER",
    "SINGLE_FINGER",
)

NUM_CLASSES: int = len(GESTURE_LABELS)

# CLASS_MAP values must be a subset of GESTURE_LABELS
# (GESTURE_LABELS may include additional custom gestures)
assert set(CLASS_MAP.values()).issubset(set(GESTURE_LABELS)), \
    f"CLASS_MAP values {set(CLASS_MAP.values())} not in GESTURE_LABELS {set(GESTURE_LABELS)}"

# ── Default configuration ───────────────────────────────────────────────────

DEFAULT_MAX_PER_CLASS: int = 5000
DEFAULT_CONFIDENCE_MIN: float = 0.7

# ── Path resolution ────────────────────────────────────────────────────────
# All default paths are relative to THIS file, not CWD.
# Running from any directory produces the same defaults.

_PACKAGE_DIR = Path(__file__).parent.resolve()


def get_training_dir() -> Path:
    """Absolute path to the training package directory."""
    return _PACKAGE_DIR


def get_output_dir(subdir: str = "data_hagrid") -> Path:
    """Resolve output directory relative to the training package."""
    return _PACKAGE_DIR / subdir


def get_raw_dir(subdir: str = "hagrid_raw") -> Path:
    """Resolve raw HaGRID image directory relative to the training package."""
    return _PACKAGE_DIR / subdir


# ── Shared feature persistence ─────────────────────────────────────────────

def save_features_by_label(
    features_by_label: Dict[str, list],
    out_dir: Path,
    *,
    log=None,
) -> int:
    """Persist extracted features to .npz files, one per label.

    Args:
        features_by_label: Mapping from gesture label string to list of
            feature vectors (each a 1-D np.ndarray of shape (70,)).
        out_dir: Directory to write .npz files into (created if needed).
        log: Logger instance for progress messages (uses module logger by default).

    Returns:
        Total number of samples saved across all labels.
    """
    _log = log or logger
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for label, feats in features_by_label.items():
        if not feats:
            _log.warning("No features collected for label %s", label)
            continue
        stacked = np.stack(feats)
        np.savez_compressed(
            out_dir / f"features_{label}.npz",
            features=stacked,
            label=label,
        )
        _log.info("Saved %s: %d samples", label, len(feats))
        total += len(feats)
    return total
