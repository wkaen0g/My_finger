"""Download HaGRID lightweight 5-class subset from HuggingFace, extract features.

The dataset is hosted at: https://huggingface.co/datasets/testdummyvt/hagRIDv2_512px_10GB
It contains ~10 GB of parquet files with embedded image bytes.
Each image is decoded, run through MediaPipe, and the 70-dim feature vector is saved.

Usage:
  python -m microgesture.training.hagrid_download [--output-dir ...] [--max-per-class ...]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from microgesture.pipeline.detector import HandDetector
from microgesture.recognition.base import extract_features
from microgesture.training._hagrid_common import (
    CLASS_MAP,
    GESTURE_LABELS,
    DEFAULT_MAX_PER_CLASS,
    get_output_dir,
    save_features_by_label,
)

logger = logging.getLogger(__name__)

REPO = "testdummyvt/hagRIDv2_512px_10GB"
REVISION = "classification"
TRAIN_FILES = [f"data/train/train-{i:05d}-of-00020.parquet" for i in range(20)]


def main(
    output_dir: str | Path | None = None,
    max_per_class: int = DEFAULT_MAX_PER_CLASS,
    repo: str = REPO,
    revision: str = REVISION,
) -> int:
    """Download HaGRID parquet files, extract 70-dim features via MediaPipe.

    Args:
        output_dir: Directory for output .npz files.
                    Defaults to <training_pkg>/data_hagrid/.
        max_per_class: Max samples per gesture class.
        repo: HuggingFace dataset repository.
        revision: Dataset revision (branch/tag/commit).

    Returns:
        Total number of samples saved.
    """
    if output_dir is None:
        output_dir = get_output_dir()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = HandDetector(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    # Initialize feature buffers using canonical label order
    by_label: dict[str, list[np.ndarray]] = {label: [] for label in GESTURE_LABELS}
    train_files = [f"data/train/train-{i:05d}-of-00020.parquet" for i in range(20)]

    for filename in tqdm(train_files, desc="Parquet files", unit="file"):
        logger.info("Downloading %s...", filename)
        try:
            path = hf_hub_download(repo, filename, revision=revision, repo_type="dataset")
        except Exception as exc:
            logger.error("Failed to download %s: %s", filename, exc)
            continue

        table = pq.read_table(path)
        labels = table["label"].to_pylist()
        images = table["image"]

        for i in tqdm(range(len(labels)), desc=f"  Images in {Path(filename).name}", unit="img", leave=False):
            label_str = labels[i]
            if label_str not in CLASS_MAP:
                continue
            mapped = CLASS_MAP[label_str]
            if len(by_label[mapped]) >= max_per_class:
                continue

            img_dict = images[i].as_py()
            img_bytes = img_dict["bytes"]
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            hand = detector.detect(frame)
            if hand is not None and hand.valid:
                by_label[mapped].append(extract_features(hand.landmarks))

        # Show progress
        counts = {lbl: len(by_label[lbl]) for lbl in GESTURE_LABELS}
        logger.info("Progress: %s", counts)

        # Early exit if all classes have enough samples
        if all(len(by_label[lbl]) >= max_per_class for lbl in GESTURE_LABELS):
            logger.info("All classes reached max_per_class=%d — stopping early.", max_per_class)
            break

    detector.close()

    total = save_features_by_label(by_label, out_dir)
    logger.info("Done → %s (%d total samples)", out_dir, total)
    return total


# ── CLI ─────────────────────────────────────────────────────────────────────

def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download HaGRID from HuggingFace and extract 70-dim gesture features.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for .npz files (default: <training_pkg>/data_hagrid/)",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=DEFAULT_MAX_PER_CLASS,
        help=f"Max samples per gesture class (default: {DEFAULT_MAX_PER_CLASS})",
    )
    parser.add_argument(
        "--repo", default=REPO,
        help=f"HuggingFace dataset repo (default: {REPO})",
    )
    parser.add_argument(
        "--revision", default=REVISION,
        help=f"Dataset revision (default: {REVISION})",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    total = main(
        output_dir=args.output_dir,
        max_per_class=args.max_per_class,
        repo=args.repo,
        revision=args.revision,
    )
    print(f"\nDone. {total} total samples saved.")
    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
