"""Free-mode data collector: record hand features with gesture labels."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from microgesture.recognition.base import GestureRecognizer, extract_features

logger = logging.getLogger(__name__)

GESTURES = ("PALM_OPEN", "FIST", "SINGLE_FINGER", "TWO_FINGER", "PINCH", "NO_HAND")


class DataCollector:
    """Records (features, label) pairs while the user performs gestures.

    Usage:
      collector = DataCollector(recognizer)
      collector.start("PALM_OPEN")
      ...  # for each frame: collector.record(landmarks)
      collector.stop()

      collector.save("training/data")
    """

    def __init__(self, recognizer: GestureRecognizer):
        self._recognizer = recognizer
        self._recording = False
        self._current_label: str = ""
        self._samples: list[tuple[np.ndarray, str]] = []
        self._start_time: float = 0.0

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def current_label(self) -> str:
        return self._current_label

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def start(self, label: str) -> None:
        if label not in GESTURES:
            raise ValueError(f"Unknown gesture: {label}. Must be one of {GESTURES}")
        self._recording = True
        self._current_label = label
        self._start_time = time.time()
        self._samples.clear()
        logger.info("Recording started: %s", label)

    def stop(self) -> None:
        self._recording = False
        elapsed = time.time() - self._start_time
        logger.info("Recording stopped: %s — %d samples in %.1fs",
                     self._current_label, self.sample_count, elapsed)
        self._current_label = ""

    def record(self, landmarks: np.ndarray) -> None:
        """Capture a frame: extract features + attach label.

        If a recognizer is provided, its prediction is stored as pseudo-label
        for later review; the *intended* label from start() is always saved
        as the ground-truth key.
        """
        if not self._recording:
            return

        features = extract_features(landmarks)
        self._samples.append((features, self._current_label))

    def save(self, directory: str | Path, merge: bool = True) -> Path:
        """Persist samples to `directory/gesture_*.npz` and metadata.json.

        Args:
            directory: Output directory.
            merge: If True, merge new samples with existing .npz files
                   (enables incremental data collection across sessions).
                   If False, overwrite existing files.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Group by label
        by_label: dict[str, list[np.ndarray]] = {}
        for features, label in self._samples:
            by_label.setdefault(label, []).append(features)

        for label, feats in by_label.items():
            stacked_new = np.stack(feats)

            if merge:
                existing_path = directory / f"features_{label}.npz"
                if existing_path.exists():
                    existing = np.load(existing_path)
                    stacked_old = existing["features"]
                    stacked = np.concatenate([stacked_old, stacked_new])
                    logger.info("Merged %s: %d + %d = %d samples",
                                label, len(stacked_old), len(stacked_new), len(stacked))
                else:
                    stacked = stacked_new
                    logger.info("Saved %s: %d samples", label, len(stacked))
            else:
                stacked = stacked_new
                logger.info("Saved %s: %d samples → %s", label, len(stacked),
                            f"features_{label}.npz")

            path = directory / f"features_{label}.npz"
            np.savez_compressed(path, features=stacked, label=label)

        # Update metadata with actual file contents
        meta = {}
        for label in GESTURES:
            path = directory / f"features_{label}.npz"
            if path.exists():
                meta[label] = np.load(path)["features"].shape[0]
            else:
                meta[label] = len(by_label.get(label, []))
        meta["_total"] = sum(meta.values())
        meta["_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        meta_path = directory / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        logger.info("Metadata → %s (total=%d)", meta_path, meta["_total"])

        self._samples.clear()
        return directory
