"""MediaPipe Hands detector wrapper using Tasks API."""

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (HandLandmarker, HandLandmarkerOptions,
                                           RunningMode)

logger = logging.getLogger(__name__)

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"


@dataclass
class HandLandmarks:
    """Normalized 21-point landmarks (x, y, z) for a single hand."""
    landmarks: np.ndarray  # shape (21, 3)
    handedness: str  # "Left" or "Right"
    confidence: float

    @property
    def valid(self) -> bool:
        return self.confidence >= 0.5


def _ensure_model(model_dir: Path | None = None) -> str:
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "hand_landmarker.task"

    if not model_path.exists():
        logger.info("Downloading hand landmarker model (~15MB)...")
        urllib.request.urlretrieve(MODEL_URL, str(model_path))
        logger.info("Model downloaded to %s", model_path)

    return str(model_path)


class HandDetector:
    """MediaPipe Hands wrapper producing normalized 21-point coordinates."""

    def __init__(self, model_path: str | None = None,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 max_num_hands: int = 1):
        if model_path is None:
            model_path = _ensure_model()

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray) -> Optional[HandLandmarks]:
        """Detect hand landmarks in BGR frame. Returns best hand or None."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None

        if len(result.hand_landmarks) == 1:
            hlm = result.hand_landmarks[0]
            handedness = result.handedness[0][0].category_name
            confidence = result.handedness[0][0].score
            if confidence < 0.5:
                return None
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hlm], dtype=np.float32)
            return HandLandmarks(landmarks=landmarks, handedness=handedness, confidence=confidence)

        # Multiple hands: pick highest-confidence hand
        best_idx = 0
        best_conf = 0.0
        for i, hc in enumerate(result.handedness):
            conf = hc[0].score
            if conf > best_conf:
                best_conf = conf
                best_idx = i

        hlm = result.hand_landmarks[best_idx]
        handedness = result.handedness[best_idx][0].category_name
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hlm], dtype=np.float32)
        return HandLandmarks(landmarks=landmarks, handedness=handedness, confidence=best_conf)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False
