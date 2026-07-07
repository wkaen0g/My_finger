"""Camera capture thread with frame buffer."""

import logging
import time
from collections import deque
from threading import Event, Thread
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraCapture:
    """Background camera capture with auto-reconnect and frame buffer."""

    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480,
                 target_fps: int = 30, buffer_size: int = 2):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self._buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[Thread] = None
        self._stop_event = Event()
        self._connected = False
        self._frame_interval = 1.0 / target_fps

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _open_camera(self) -> bool:
        try:
            self._cap = cv2.VideoCapture(self.device_id)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            if not self._cap.isOpened():
                logger.error("Cannot open camera %d", self.device_id)
                return False
            self._connected = True
            logger.info("Camera %d opened: %dx%d", self.device_id, self.width, self.height)
            return True
        except Exception:
            logger.exception("Camera open error")
            return False

    def _close_camera(self) -> None:
        self._connected = False
        if self._cap:
            self._cap.release()
            self._cap = None

    def latest_frame(self) -> Optional[np.ndarray]:
        if self._buffer:
            return self._buffer[-1]
        return None

    def _run(self) -> None:
        reconnect_delay = 0.5
        max_delay = 8.0
        attempts = 0

        while not self._stop_event.is_set():
            if not self._connected:
                logger.info("Camera reconnect attempt %d in %.1fs",
                            attempts + 1, reconnect_delay)
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_delay)
                if self._open_camera():
                    logger.info("Camera reconnected after %d attempts", attempts + 1)
                    reconnect_delay = 0.5
                    attempts = 0
                else:
                    attempts += 1
                continue

            loop_start = time.perf_counter()
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Frame read failed, camera may be lost")
                self._close_camera()
                attempts += 1
                continue

            frame = cv2.flip(frame, 1)
            self._buffer.append(frame)
            attempts = 0

            elapsed = time.perf_counter() - loop_start
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True, name="camera-capture")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._close_camera()
        logger.info("Camera capture stopped")
