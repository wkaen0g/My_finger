"""Main entry point for the microgesture application."""

import logging
import logging.handlers
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyautogui

from .config import get_config
from .pipeline.air_tap import AirTapDetector, TapResult
from .pipeline.capture import CameraCapture
from .pipeline.cursor import CursorController
from .pipeline.detector import HandDetector
from .pipeline.gesture_engine import Gesture, RuleEngine
from .pipeline.pinch import PinchDetector
from .pipeline.scroll import ScrollDetector
from .system.input import InputController
from .system.tray import SystemTray, TrayCallbacks, TrayState

logger = logging.getLogger(__name__)


def setup_logging(config) -> None:
    log_dir = Path(config.get("logging", "dir", default="logs"))
    if not log_dir.is_absolute():
        log_dir = Path(__file__).parent / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "microgesture.log"

    level = getattr(logging, config.get("logging", "level", default="INFO"), logging.INFO)
    max_bytes = config.get("logging", "max_bytes", default=5_242_880)
    backup_count = config.get("logging", "backup_count", default=3)

    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(ch)


class GesturePipeline:
    """Orchestrates capture, detection, gesture logic, and system events."""

    def __init__(self, config, on_status_change=None):
        self.config = config
        self._running = False
        self._tracking = True
        self._thread: threading.Thread | None = None
        self._on_status_change = on_status_change
        self._last_camera_ok = False

        # Components
        self.capture = CameraCapture(
            device_id=config.get("camera", "device_id", default=0),
            width=config.get("camera", "width", default=640),
            height=config.get("camera", "height", default=480),
            target_fps=config.get("camera", "target_fps", default=30),
        )
        self.detector = HandDetector(
            min_detection_confidence=config.get("mediapipe", "min_detection_confidence", default=0.5),
            min_tracking_confidence=config.get("mediapipe", "min_tracking_confidence", default=0.5),
        )
        self.engine = RuleEngine(
            tip_mcp_open_threshold=config.get("gesture", "tip_mcp_open_threshold", default=0.25),
            tip_mcp_fist_threshold=config.get("gesture", "tip_mcp_fist_threshold", default=0.12),
            pinch_threshold_ratio=config.get("gesture", "pinch_threshold_ratio", default=0.35),
        )
        self.tap = AirTapDetector(
            tap_threshold=config.get("gesture", "tap_threshold", default=0.3),
            min_bend=config.get("gesture", "tap_min_bend", default=0.15),
            suppress_threshold=config.get("gesture", "tap_suppress_threshold", default=0.1),
            bend_timeout=config.get("gesture", "tap_bend_timeout", default=12),
            rebound_timeout=config.get("gesture", "tap_rebound_timeout", default=8),
            cooldown_frames=config.get("gesture", "tap_cooldown_frames", default=8),
        )
        self._tap_result: TapResult | None = None
        self.pinch = PinchDetector(
            pinch_threshold_ratio=config.get("gesture", "pinch_threshold_ratio", default=0.35),
        )
        import pyautogui
        sw, sh = pyautogui.size()
        self.cursor = CursorController(
            screen_width=sw, screen_height=sh,
            sensitivity=config.get("cursor", "sensitivity", default=0.6),
            deadzone=config.get("cursor", "deadzone", default=0.003),
            beta=config.get("cursor", "smoothing_beta", default=0.007),
            fcmin=config.get("cursor", "smoothing_fcmin", default=1.0),
            min_cutoff=config.get("cursor", "smoothing_cutoff", default=1.0),
        )
        self.scroll = ScrollDetector(
            screen_height=sh,
            sensitivity=config.get("scroll", "sensitivity", default=40.0),
            beta=config.get("scroll", "smoothing_beta", default=0.007),
            fcmin=config.get("scroll", "smoothing_fcmin", default=1.0),
            min_cutoff=config.get("scroll", "smoothing_cutoff", default=1.0),
        )
        self.input_ctrl = InputController()
        self._no_hand_start: float | None = None
        self._fist_start: float | None = None
        self._prev_gesture = Gesture.NO_HAND
        self._right_click_mode = config.get("system", "right_click_mode", default="fist_tap")
        self._preview = config.get("debug", "preview_window", default=False)
        self._preview_frame_count = 0

    def _draw_preview(self, frame, hand, gesture) -> None:
        if not self._preview:
            return
        self._preview_frame_count += 1
        if self._preview_frame_count % 2 != 0:  # show every 2nd frame
            return

        import cv2
        h, w = frame.shape[:2]
        display = frame.copy()

        if hand is not None and gesture is not None:
            # Draw landmarks
            for i, (x, y, z) in enumerate(hand.landmarks):
                px, py = int(x * w), int(y * h)
                cv2.circle(display, (px, py), 3, (0, 255, 0), -1)
                cv2.putText(display, str(i), (px + 4, py), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (255, 255, 255), 1)

            # Gesture label
            color = (0, 255, 0)
            if gesture.gesture == Gesture.FIST:
                color = (0, 0, 255)
            elif gesture.gesture == Gesture.TWO_FINGER:
                color = (255, 0, 0)
            elif gesture.gesture == Gesture.PINCH:
                color = (255, 255, 0)
            label = f"{gesture.gesture.name} | drag={self.input_ctrl.is_dragging}"
            cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)

            # Pinch distance
            thumb, idx = hand.landmarks[4], hand.landmarks[8]
            pinch = np.linalg.norm(thumb - idx) / (np.linalg.norm(hand.landmarks[0] - hand.landmarks[9]) + 1e-6)

            # Tap ratio (finger bend detection)
            tap_mcp = hand.landmarks[5]
            tap_pip = hand.landmarks[6]
            tap_tip = hand.landmarks[8]
            tap_ratio = (np.linalg.norm(tap_tip - tap_mcp)
                         / (np.linalg.norm(tap_pip - tap_mcp) + 1e-6))

            dratio_str = ""
            if self._tap_result is not None:
                dratio_str = f" 变化={self._tap_result.dratio:+.3f}"
                if self._tap_result.suppress_cursor:
                    dratio_str += " 抑制"

            cv2.putText(display, f"捏合={pinch:.3f} 弯曲比={tap_ratio:.2f}{dratio_str}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Tip-MCP distances
            TIP = (4, 8, 12, 16, 20)
            MCP = (2, 5, 9, 13, 17)
            names = ("Th", "Ix", "Md", "Rn", "Pk")
            for i, (t, m, n) in enumerate(zip(TIP, MCP, names)):
                d = np.linalg.norm(hand.landmarks[t] - hand.landmarks[m])
                cv2.putText(display, f"{n}={d:.3f}", (10, 80 + i * 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        else:
            cv2.putText(display, "NO HAND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

        cv2.imshow("MicroGesture Preview", display)
        cv2.waitKey(1)

    def _process_frame(self) -> None:
        frame = self.capture.latest_frame()
        if frame is None:
            return

        hand = self.detector.detect(frame)
        if hand is None:
            self._handle_no_hand()
            self._draw_preview(frame, None, None)
            return

        self._no_hand_start = None
        self._tap_result = self.tap.update(hand.landmarks)
        gesture = self.engine.classify(hand.landmarks)

        # Handle gesture transitions
        if gesture.gesture == Gesture.PALM_OPEN or gesture.gesture == Gesture.PINCH:
            self._fist_start = None

        # Dispatch by gesture
        if gesture.gesture == Gesture.PALM_OPEN:
            self.cursor.unfreeze()
            self._handle_cursor_move(hand.landmarks)
            self._handle_tap()
            self._handle_pinch(hand.landmarks)

        elif gesture.gesture == Gesture.FIST:
            self.cursor.freeze()
            if self._fist_start is None:
                self._fist_start = time.time()
            if self._right_click_mode == "fist_tap":
                self._handle_tap(right_click=True)

        elif gesture.gesture == Gesture.TWO_FINGER:
            self.cursor.freeze()
            self._handle_scroll(hand.landmarks)
            if self._right_click_mode == "two_finger":
                self._handle_tap(right_click=True)

        elif gesture.gesture == Gesture.PINCH:
            self.cursor.unfreeze()
            self._handle_cursor_move(hand.landmarks)
            self._handle_pinch(hand.landmarks)

        if self._prev_gesture == Gesture.TWO_FINGER and gesture.gesture != Gesture.TWO_FINGER:
            self.scroll.stop()

        self._draw_preview(frame, hand, gesture)
        self._prev_gesture = gesture.gesture

    def _handle_no_hand(self) -> None:
        timeout = self.config.get("system", "no_hand_sleep_timeout", default=5.0)
        if self._no_hand_start is None:
            self._no_hand_start = time.time()
        elif timeout > 0 and (time.time() - self._no_hand_start) > timeout:
            self.cursor.freeze()
            self.scroll.stop()

    def _handle_cursor_move(self, landmarks) -> None:
        # Suppress cursor when finger is bending (tap motion)
        if self._tap_result is not None and self._tap_result.suppress_cursor:
            return
        tip = landmarks[8]
        dx, dy = self.cursor.update(tip[0], tip[1])
        self.input_ctrl.move(dx, dy)

    def _handle_tap(self, right_click: bool = False) -> None:
        if self._tap_result is None or self._tap_result.event is None:
            return
        if self.input_ctrl.is_dragging:
            return
        if right_click:
            self.input_ctrl.right_click()
        else:
            self.input_ctrl.click()

    def _handle_pinch(self, landmarks) -> None:
        event = self.pinch.update(landmarks)
        if event:
            from .pipeline.pinch import PinchState
            if event.state == PinchState.PINCHING:
                self.input_ctrl.drag_start()
            else:
                self.input_ctrl.drag_end()

    def _handle_scroll(self, landmarks) -> None:
        if self._prev_gesture != Gesture.TWO_FINGER:
            self.scroll.start()
        delta = self.scroll.update(landmarks[8], landmarks[12])
        if delta != 0:
            self.input_ctrl.scroll(int(round(delta)))

    def _run_loop(self) -> None:
        while self._running:
            try:
                if self._tracking:
                    self._process_frame()
            except pyautogui.FailSafeException:
                logger.warning("Failsafe triggered — cursor hit screen corner, freezing")
                self.cursor.freeze()
                self._tracking = False
            except Exception:
                logger.exception("Pipeline error, continuing")

            camera_ok = self.capture.is_connected
            if camera_ok != self._last_camera_ok:
                self._last_camera_ok = camera_ok
                if self._on_status_change:
                    self._on_status_change(camera_ok)
            time.sleep(0.001)

    def toggle_tracking(self) -> None:
        self._tracking = not self._tracking
        logger.info("Tracking %s", "enabled" if self._tracking else "disabled")

    def set_sensitivity(self, value: float) -> None:
        self.cursor.sensitivity = value
        logger.info("Sensitivity set to %.1f", value)

    def set_right_click(self, mode: str) -> None:
        self._right_click_mode = mode
        logger.info("Right click mode set to %s", mode)

    def start(self) -> None:
        self._running = True
        self.capture.start()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="pipeline")
        self._thread.start()
        logger.info("Gesture pipeline started")

    def stop(self) -> None:
        self._running = False
        self.capture.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self.detector.close()
        logger.info("Gesture pipeline stopped")


def main() -> None:
    config = get_config()
    setup_logging(config)

    logger.info("MicroGesture v0.1.0 starting")

    pipeline = GesturePipeline(config)

    tray = SystemTray(TrayCallbacks(
        toggle_tracking=pipeline.toggle_tracking,
        set_sensitivity=pipeline.set_sensitivity,
        set_right_click=pipeline.set_right_click,
        quit_app=lambda: shutdown(pipeline, tray),
    ))

    pipeline._on_status_change = lambda c: tray.set_state(
        TrayState.NORMAL if c else TrayState.NO_CAMERA
    )

    pipeline.start()

    def _tray_thread():
        tray.run()

    tt = threading.Thread(target=_tray_thread, daemon=True, name="tray")
    tt.start()

    # Give camera a moment to connect, then set initial state
    for _ in range(30):
        time.sleep(0.1)
        if pipeline.capture.is_connected:
            break
    tray.set_state(TrayState.NORMAL if pipeline.capture.is_connected else TrayState.NO_CAMERA)

    try:
        while tt.is_alive():
            tt.join(1)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown(pipeline, tray)


def shutdown(pipeline: GesturePipeline, tray: SystemTray) -> None:
    pipeline.stop()
    tray.stop()
    logger.info("MicroGesture shutdown complete")


if __name__ == "__main__":
    main()
