"""HaGRID dataset: extract 70-dim features via MediaPipe from local images.

Usage:
  1. Download HaGRID subset (5 classes) from:
     https://github.com/hukenovs/hagrid/releases
     Get: palm.zip, fist.zip, peace.zip, pinch.zip, no_gesture.zip
     Extract each into: microgesture/training/hagrid_raw/{class}/

  2. Run: python -m microgesture.training.hagrid_loader [--raw-dir ...] [--output-dir ...]

  3. Output: data_hagrid/features_*.npz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from microgesture.pipeline.detector import HandDetector
from microgesture.recognition.base import extract_features
from microgesture.training._hagrid_common import (
    CLASS_MAP,
    DEFAULT_CONFIDENCE_MIN,
    DEFAULT_MAX_PER_CLASS,
    get_output_dir,
    get_raw_dir,
    save_features_by_label,
)

logger = logging.getLogger(__name__)


def process_hagrid(
    raw_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    max_per_class: int = DEFAULT_MAX_PER_CLASS,
    confidence_min: float = DEFAULT_CONFIDENCE_MIN,
) -> int:
    """Process HaGRID images through MediaPipe, extract features, save .npz.

    Args:
        raw_dir: Directory containing HaGRID class subdirectories.
                 Defaults to <training_pkg>/hagrid_raw/.
        out_dir: Directory for output .npz files.
                 Defaults to <training_pkg>/data_hagrid/.
        max_per_class: Max images to process per gesture class.
        confidence_min: Minimum MediaPipe detection confidence.

    Returns:
        Total number of samples saved.
    """
    if raw_dir is None:
        raw_dir = get_raw_dir()
    if out_dir is None:
        out_dir = get_output_dir()

    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = HandDetector(
        min_detection_confidence=confidence_min,
        min_tracking_confidence=confidence_min,
    )
    total = 0

    try:
        for hagrid_name, label in CLASS_MAP.items():
            class_dir = raw_dir / hagrid_name
            if not class_dir.is_dir():
                logger.warning("Skipping %s — directory not found at %s", label, class_dir)
                continue

            images = sorted(class_dir.glob("*.*"))[:max_per_class]
            if not images:
                logger.warning("No images found for %s in %s", label, class_dir)
                continue

            features_list: list[np.ndarray] = []

            for img_path in tqdm(images, desc=f"Processing {label}", unit="img"):
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue
                hand = detector.detect(frame)
                if hand is None:
                    continue

                features_list.append(extract_features(hand.landmarks))

            if features_list:
                stacked = np.stack(features_list)
                np.savez_compressed(
                    out_dir / f"features_{label}.npz",
                    features=stacked,
                    label=label,
                )
                logger.info("%s: %d samples saved", label, len(features_list))
                total += len(features_list)
            else:
                logger.warning("%s: no valid hand detections — skipping", label)
    finally:
        detector.close()

    logger.info("HaGRID processing complete: %d total samples → %s", total, out_dir)
    return total


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Process local HaGRID images through MediaPipe to extract gesture features.",
    )
    parser.add_argument(
        "--raw-dir", default=None,
        help="Directory with HaGRID class subdirectories (default: <training_pkg>/hagrid_raw/)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for .npz files (default: <training_pkg>/data_hagrid/)",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=DEFAULT_MAX_PER_CLASS,
        help=f"Max images per gesture class (default: {DEFAULT_MAX_PER_CLASS})",
    )
    parser.add_argument(
        "--confidence-min", type=float, default=DEFAULT_CONFIDENCE_MIN,
        help=f"Minimum MediaPipe detection confidence (default: {DEFAULT_CONFIDENCE_MIN})",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    total = process_hagrid(
        raw_dir=args.raw_dir,
        out_dir=args.output_dir,
        max_per_class=args.max_per_class,
        confidence_min=args.confidence_min,
    )
    print(f"\nDone. {total} total samples saved.")
    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
