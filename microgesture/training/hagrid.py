"""Unified data preparation CLI (HaGRID + 20BN-Jester V1).

Usage:
  # Download HaGRID from HuggingFace and extract features (automated, ~10 GB)
  python -m microgesture.training.hagrid download [--max-per-class 5000]

  # Process locally-downloaded HaGRID images (offline mode)
  python -m microgesture.training.hagrid process [--raw-dir hagrid_raw/] [--max-per-class 2000]

  # Process 20BN-Jester V1 frames with pseudo-labeling
  python -m microgesture.training.hagrid jester [--frames-dir ...] [--max-per-class 5000]
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare HaGRID gesture features for classifier pretraining.",
    )
    sub = parser.add_subparsers(dest="command")

    # ── download ────────────────────────────────────────────────────────
    dl = sub.add_parser("download", help="Download from HuggingFace and extract features")
    dl.add_argument(
        "--output-dir", default=None,
        help="Directory for output .npz files (default: <training_pkg>/data_hagrid/)",
    )
    dl.add_argument(
        "--max-per-class", type=int, default=5000,
        help="Max samples per gesture class (default: 5000)",
    )
    dl.add_argument(
        "--repo", default=None,
        help="Override HuggingFace dataset repo",
    )
    dl.add_argument(
        "--revision", default="classification",
        help="Dataset revision (default: classification)",
    )

    # ── process ─────────────────────────────────────────────────────────
    proc = sub.add_parser("process", help="Process local pre-downloaded HaGRID images")
    proc.add_argument(
        "--raw-dir", default=None,
        help="Directory containing HaGRID class subdirectories (default: <training_pkg>/hagrid_raw/)",
    )
    proc.add_argument(
        "--output-dir", default=None,
        help="Directory for output .npz files (default: <training_pkg>/data_hagrid/)",
    )
    proc.add_argument(
        "--max-per-class", type=int, default=2000,
        help="Max samples per gesture class (default: 2000)",
    )
    proc.add_argument(
        "--confidence-min", type=float, default=0.7,
        help="Minimum MediaPipe detection confidence (default: 0.7)",
    )

    # ── jester ────────────────────────────────────────────────────────────
    jester = sub.add_parser("jester", help="Process 20BN-Jester V1 frames with pseudo-labeling")
    jester.add_argument(
        "--frames-dir", default=None,
        help="Directory containing Jester video-id subdirectories with .jpg frames "
             "(default: D:/20BN-Jester V1/downloads/20bn-jester-v1/)",
    )
    jester.add_argument(
        "--output-dir", default=None,
        help="Directory for output .npz files (default: <training_pkg>/data_jester/)",
    )
    jester.add_argument(
        "--max-per-class", type=int, default=5000,
        help="Max samples per gesture class (default: 5000)",
    )
    jester.add_argument(
        "--confidence-min", type=float, default=0.7,
        help="Minimum MediaPipe hand detection confidence (default: 0.7)",
    )
    jester.add_argument(
        "--frames-per-video", type=int, default=3,
        help="Frames to sample per Jester video (default: 3)",
    )
    jester.add_argument(
        "--pseudo-confidence-min", type=float, default=0.5,
        help="Minimum pseudo-label confidence to accept (default: 0.5)",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.command == "download":
        from microgesture.training.hagrid_download import main as download_main

        kwargs = {
            "output_dir": args.output_dir,
            "max_per_class": args.max_per_class,
        }
        if args.repo:
            kwargs["repo"] = args.repo
        kwargs["revision"] = args.revision

        total = download_main(**kwargs)
        dest = args.output_dir or "(default)"
        print(f"\nDownload complete: {total} samples saved to {dest}")

    elif args.command == "process":
        from microgesture.training.hagrid_loader import process_hagrid

        total = process_hagrid(
            raw_dir=args.raw_dir,
            out_dir=args.output_dir,
            max_per_class=args.max_per_class,
            confidence_min=args.confidence_min,
        )
        dest = args.output_dir or "(default)"
        print(f"\nProcessing complete: {total} samples saved to {dest}")

    elif args.command == "jester":
        from microgesture.training.jester_loader import process_jester

        total = process_jester(
            frames_dir=args.frames_dir,
            out_dir=args.output_dir,
            max_per_class=args.max_per_class,
            confidence_min=args.confidence_min,
            frames_per_video=args.frames_per_video,
            pseudo_confidence_min=args.pseudo_confidence_min,
        )
        dest = args.output_dir or "(default)"
        print(f"\nJester processing complete: {total} samples saved to {dest}")

    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
