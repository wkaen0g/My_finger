"""Guided data collector: on-screen instructions → auto-record → save.

Uses PIL for Chinese text rendering (OpenCV putText does not support CJK).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from microgesture.config import get_config
from microgesture.pipeline.capture import CameraCapture
from microgesture.pipeline.detector import HandDetector
from microgesture.recognition.base import extract_features
from microgesture.training.data_collector import DataCollector, GESTURES

logger = logging.getLogger(__name__)

_GESTURE_CN = {
    "PALM_OPEN":  ("手掌张开", (0, 255, 0)),
    "FIST":       ("握拳",       (0, 0, 255)),
    "TWO_FINGER": ("双指伸出", (255, 0, 0)),
    "PINCH":      ("捏合",       (255, 255, 0)),
    "NO_HAND":    ("手移出画面", (128, 128, 128)),
}

_FRAMES_PER_GESTURE = 500
_READY_SECONDS = 10


def _pil_text(img: np.ndarray, text: str, xy, font_size: int,
              color, anchor="mm") -> np.ndarray:
    """Draw CJK-compatible text on a numpy image using PIL."""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("simhei.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("msyh.ttc", font_size)
        except OSError:
            font = ImageFont.load_default()
    draw.text(xy, text, font=font, fill=color[::-1], anchor=anchor)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _draw_instruction(frame: np.ndarray, text: str, color,
                      count_text: str | None, stage: str) -> np.ndarray:
    """Draw full-screen instruction overlay. Returns modified frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
    blended = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    # Stage (top)
    blended = _pil_text(blended, stage, (w // 2, h // 2 - 120), 36, color)

    # Gesture name (center, large)
    blended = _pil_text(blended, text, (w // 2, h // 2 - 30), 48, color)

    # Countdown / frame count
    if count_text is not None:
        blended = _pil_text(blended, count_text, (w // 2, h // 2 + 60),
                            64, (255, 255, 255))

    return blended


def run_guided_collection(data_dir: str | Path = "microgesture/training/data",
                          frames_per_gesture: int = _FRAMES_PER_GESTURE):
    """Run the guided collection flow.

    Args:
        data_dir: Output directory for .npz files.
        frames_per_gesture: Number of frames to record per gesture class.
                            Default 500; increase for more training data.
    """
    config = get_config()
    data_dir = Path(data_dir)

    capture = CameraCapture(
        device_id=config.get("camera", "device_id", default=0),
        width=config.get("camera", "width", default=640),
        height=config.get("camera", "height", default=480),
        target_fps=config.get("camera", "target_fps", default=30),
    )
    detector = HandDetector(
        min_detection_confidence=config.get("mediapipe", "min_detection_confidence", default=0.5),
        min_tracking_confidence=config.get("mediapipe", "min_tracking_confidence", default=0.5),
    )
    collector = DataCollector(None)

    capture.start()
    time.sleep(1.0)

    cv2.namedWindow("Gesture Collector", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Gesture Collector", cv2.WND_PROP_TOPMOST, 1)

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)

    for label in GESTURES:
        name, color = _GESTURE_CN[label]

        # ── Ready phase ───────────────────────────────────────────
        for sec in range(_READY_SECONDS, 0, -1):
            frame = capture.latest_frame()
            if frame is None:
                frame = dummy.copy()
            display = _draw_instruction(frame, name, color, str(sec), "准备")
            cv2.imshow("Gesture Collector", display)
            cv2.waitKey(1)
            time.sleep(1.0)

        # ── Recording phase ───────────────────────────────────────
        collector.start(label)
        total_iterations = 0
        total_frames = 0
        total_detections = 0
        max_iterations = frames_per_gesture * 15

        while collector.sample_count < frames_per_gesture:
            total_iterations += 1
            if total_iterations > max_iterations:
                logger.warning("%s: timeout after %d iters (frames=%d det=%d samples=%d)",
                               name, total_iterations, total_frames,
                               total_detections, collector.sample_count)
                break

            frame = capture.latest_frame()
            if frame is None:
                time.sleep(0.003)
                continue
            total_frames += 1

            if label == "NO_HAND":
                collector.record(np.zeros((21, 3), dtype=np.float32))
            else:
                hand = detector.detect(frame)
                if hand is not None:
                    collector.record(hand.landmarks)
                    total_detections += 1

            # Update display every 30 iterations
            if total_iterations % 30 == 0 or collector.sample_count == frames_per_gesture:
                info = f"{collector.sample_count}/{frames_per_gesture}"
                display = _draw_instruction(frame, name, color, info, "录制中")
                cv2.imshow("Gesture Collector", display)
                cv2.waitKey(1)

        collector.stop()
        logger.info("%s: %d frames (%d iters)", name, collector.sample_count,
                     total_iterations)
        collector.save(data_dir)

    # ── Save ───────────────────────────────────────────────────────
    dummy.fill(0)
    dummy = _pil_text(dummy, "保存中...",
                      (640 // 2, 480 // 2), 40, (0, 255, 0))
    cv2.imshow("Gesture Collector", dummy)
    cv2.waitKey(1)

    capture.stop()
    detector.close()
    cv2.destroyAllWindows()
    logger.info("Guided collection complete → %s (%d frames/gesture)",
                data_dir, frames_per_gesture)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Guided gesture data collector")
    parser.add_argument("--data-dir", default="microgesture/training/data",
                        help="Output directory for collected .npz files "
                             "(default: microgesture/training/data)")
    parser.add_argument("--frames", type=int, default=_FRAMES_PER_GESTURE,
                        help=f"Frames per gesture class (default: {_FRAMES_PER_GESTURE})")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_guided_collection(args.data_dir, frames_per_gesture=args.frames)
