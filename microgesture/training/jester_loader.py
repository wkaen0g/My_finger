"""20BN-Jester V1 dataset: pseudo-label frames via MediaPipe + RuleEngine.

Usage:
  1. Ensure Jester frames are extracted to:
     D:/20BN-Jester V1/downloads/20bn-jester-v1/{video_id}/#####.jpg

  2. Run: python -m microgesture.training.jester_loader [--frames-dir ...] [--output-dir ...]

  3. Output: data_jester/features_*.npz (5-class pseudo-labeled features)

Strategy:
  - Sample 1-3 frames per video (evenly spaced)
  - MediaPipe Hands detects landmarks (confidence >= 0.7)
  - RuleEngine pseudo-labels each frame into one of the 4 hand-shape classes
  - Frames with fallback (low-confidence) pseudo-labels are skipped
  - NO_HAND is NOT collected from Jester (self-collected data covers it)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from microgesture.pipeline.detector import HandDetector
from microgesture.pipeline.gesture_engine import Gesture, RuleEngine
from microgesture.recognition.base import extract_features
from microgesture.training._hagrid_common import (
    DEFAULT_CONFIDENCE_MIN,
    DEFAULT_MAX_PER_CLASS,
    GESTURE_LABELS,
    get_output_dir,
    save_features_by_label,
)

logger = logging.getLogger(__name__)

# ── Jester-specific defaults ───────────────────────────────────────────────────

JESTER_FRAMES_DIR = Path("D:/20BN-Jester V1/downloads/20bn-jester-v1")
FRAMES_PER_VIDEO = 3          # frames sampled per video
PSEUDO_CONFIDENCE_MIN = 0.7   # skip pseudo-labels below this confidence

# Gesture enum → canonical label string (subset used for pseudo-labeling)
_GESTURE_ENUM_TO_LABEL = {
    Gesture.PALM_OPEN: "PALM_OPEN",
    Gesture.FIST: "FIST",
    Gesture.TWO_FINGER: "TWO_FINGER",
    Gesture.PINCH: "PINCH",
}


def _sample_frame_indices(n_frames: int, n_samples: int) -> list[int]:
    """Choose evenly-spaced frame indices from a video.

    Avoids the very first and very last frame (which may be blurrier)
    by distributing indices across the interior of the sequence.

    Args:
        n_frames: Total number of .jpg frames in this video folder.
        n_samples: Desired number of frames to sample.

    Returns:
        List of 0-based frame indices, sorted.
    """
    if n_frames <= n_samples:
        return list(range(n_frames))
    # Distribute n_samples indices evenly across [0, n_frames)
    step = n_frames / (n_samples + 1)
    indices = [int(step * (i + 1)) for i in range(n_samples)]
    # Clamp to valid range (just in case of off-by-one)
    return [min(i, n_frames - 1) for i in indices]


def process_jester(
    frames_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    max_per_class: int = DEFAULT_MAX_PER_CLASS,
    confidence_min: float = DEFAULT_CONFIDENCE_MIN,
    frames_per_video: int = FRAMES_PER_VIDEO,
    pseudo_confidence_min: float = PSEUDO_CONFIDENCE_MIN,
) -> int:
    """Process Jester frames through MediaPipe + RuleEngine, save pseudo-labeled features.

    Args:
        frames_dir: Root directory containing video-id subdirectories with .jpg frames.
                    Defaults to D:/20BN-Jester V1/downloads/20bn-jester-v1/.
        out_dir: Directory for output .npz files.
                 Defaults to <training_pkg>/data_jester/.
        max_per_class: Max samples to collect per gesture class (default: 5000).
        confidence_min: Minimum MediaPipe hand detection confidence (default: 0.7).
        frames_per_video: Number of frames to sample from each video (default: 3).
        pseudo_confidence_min: Minimum RuleEngine confidence to accept pseudo-label (default: 0.5).
                               Labels with confidence below this are treated as "uncertain" and skipped.

    Returns:
        Total number of samples saved.
    """
    if frames_dir is None:
        frames_dir = JESTER_FRAMES_DIR
    if out_dir is None:
        out_dir = get_output_dir("data_jester")

    frames_dir = Path(frames_dir)
    out_dir = Path(out_dir)

    if not frames_dir.is_dir():
        logger.error("Jester frames directory not found: %s", frames_dir)
        return 0

    # ── Discover video folders ───────────────────────────────────────────
    video_dirs = sorted(
        d for d in frames_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    if not video_dirs:
        logger.error("No video-id subdirectories found in %s", frames_dir)
        return 0

    logger.info("Found %d video folders in %s", len(video_dirs), frames_dir)

    # ── Initialise components ─────────────────────────────────────────────
    detector = HandDetector(
        min_detection_confidence=confidence_min,
        min_tracking_confidence=confidence_min,
    )
    rule_engine = RuleEngine()

    # Per-label feature buffers (all 5 gesture classes, including NO_HAND)
    target_labels = ["PALM_OPEN", "FIST", "TWO_FINGER", "PINCH", "NO_HAND"]
    features_by_label: dict[str, list[np.ndarray]] = {l: [] for l in target_labels}
    class_full: dict[str, bool] = {l: False for l in target_labels}

    # Pre-allocated zero landmarks for NO_HAND frames (avoids repeated allocation)
    _ZERO_LANDMARKS = np.zeros((21, 3), dtype=np.float32)

    total_processed = 0
    total_detected = 0

    # ── Main loop ─────────────────────────────────────────────────────────
    for vid_dir in tqdm(video_dirs, desc="Processing Jester videos", unit="vid"):
        # Skip if all classes are already full
        if all(class_full.values()):
            logger.info("All classes reached max_per_class=%d — stopping early", max_per_class)
            break

        frame_paths = sorted(vid_dir.glob("*.jpg"))
        if not frame_paths:
            continue

        indices = _sample_frame_indices(len(frame_paths), frames_per_video)

        for idx in indices:
            total_processed += 1

            # Read frame
            frame = cv2.imread(str(frame_paths[idx]))
            if frame is None:
                continue

            # MediaPipe detection
            hand = detector.detect(frame)
            if hand is None:
                # No hand detected → record as NO_HAND
                if not class_full["NO_HAND"]:
                    feat = extract_features(_ZERO_LANDMARKS)
                    features_by_label["NO_HAND"].append(feat)
                    if len(features_by_label["NO_HAND"]) >= max_per_class:
                        class_full["NO_HAND"] = True
                        logger.info("✓ NO_HAND: reached %d samples (max)", len(features_by_label["NO_HAND"]))
                continue

            total_detected += 1

            # Rule-engine pseudo-label
            result = rule_engine.classify(hand.landmarks)
            label = _GESTURE_ENUM_TO_LABEL.get(result.gesture)
            if label is None:
                continue  # unknown gesture — skip

            # Filter low-confidence pseudo-labels (fallback PALM_OPEN has conf=0.3)
            if result.confidence < pseudo_confidence_min:
                continue

            # Check if class still needs samples
            if class_full[label]:
                continue

            # Extract feature vector
            feat = extract_features(hand.landmarks)
            features_by_label[label].append(feat)

            # Check if class just reached max
            if len(features_by_label[label]) >= max_per_class:
                class_full[label] = True
                logger.info("✓ %s: reached %d samples (max)", label, len(features_by_label[label]))

    detector.close()

    # ── Report stats ──────────────────────────────────────────────────────
    logger.info(
        "Processing stats: %d frames read, %d hands detected (%.1f%%), %d samples collected",
        total_processed, total_detected,
        100 * total_detected / max(total_processed, 1),
        sum(len(v) for v in features_by_label.values()),
    )
    for label in target_labels:
        logger.info("  %s: %d samples", label, len(features_by_label[label]))

    # ── Persist ───────────────────────────────────────────────────────────
    total = save_features_by_label(features_by_label, out_dir)
    logger.info("Jester processing complete: %d total samples → %s", total, out_dir)
    return total


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Process 20BN-Jester V1 frames through MediaPipe + RuleEngine "
                    "to extract pseudo-labeled gesture features.",
    )
    parser.add_argument(
        "--frames-dir", default=None,
        help="Directory containing Jester video-id subdirectories with .jpg frames "
             "(default: D:/20BN-Jester V1/downloads/20bn-jester-v1/)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for .npz files (default: <training_pkg>/data_jester/)",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=DEFAULT_MAX_PER_CLASS,
        help=f"Max samples per gesture class (default: {DEFAULT_MAX_PER_CLASS})",
    )
    parser.add_argument(
        "--confidence-min", type=float, default=DEFAULT_CONFIDENCE_MIN,
        help=f"Minimum MediaPipe hand detection confidence (default: {DEFAULT_CONFIDENCE_MIN})",
    )
    parser.add_argument(
        "--frames-per-video", type=int, default=FRAMES_PER_VIDEO,
        help=f"Frames to sample per Jester video (default: {FRAMES_PER_VIDEO})",
    )
    parser.add_argument(
        "--pseudo-confidence-min", type=float, default=PSEUDO_CONFIDENCE_MIN,
        help=f"Minimum pseudo-label confidence to accept (default: {PSEUDO_CONFIDENCE_MIN})",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    total = process_jester(
        frames_dir=args.frames_dir,
        out_dir=args.output_dir,
        max_per_class=args.max_per_class,
        confidence_min=args.confidence_min,
        frames_per_video=args.frames_per_video,
        pseudo_confidence_min=args.pseudo_confidence_min,
    )
    print(f"\nDone. {total} total samples saved.")
    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
