"""Main entry point for the microgesture application."""

import argparse
import logging
import logging.handlers
import os
import signal
import sys
import threading
import time
from pathlib import Path

import ctypes

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import get_config
from .pipeline.air_tap import AirTapDetector, TapResult
from .pipeline.capture import CameraCapture
from .pipeline.cursor import CursorController
from .pipeline.detector import HandDetector
from .pipeline.gesture_engine import Gesture, RuleEngine
from .pipeline.pinch import PinchDetector
from .pipeline.scroll import ScrollDetector
from .recognition.base import GestureRecognizer
from .recognition.static_classifier import StaticClassifier
from .system.input import InputController
from .system.tray import SystemTray, TrayCallbacks, TrayState

logger = logging.getLogger(__name__)

_GESTURE_CN = {
    "PALM_OPEN": "张开",
    "FIST": "握拳",
    "TWO_FINGER": "双指",
    "PINCH": "捏合",
    "NO_HAND": "无手",
}


def _cjk_text(img: np.ndarray, text: str, xy, font_size: int,
              color, anchor="lt") -> np.ndarray:
    """Draw CJK text on BGR numpy image using PIL."""
    import cv2

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


def setup_logging(config, debug: bool = False) -> None:
    log_dir = Path(config.get("logging", "dir", default="logs"))
    if not log_dir.is_absolute():
        log_dir = Path(__file__).parent / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "microgesture.log"

    level_name = "DEBUG" if debug else config.get("logging", "level", default="INFO")
    level = getattr(logging, level_name, logging.INFO)
    max_bytes = config.get("logging", "max_bytes", default=10_485_760)
    backup_count = config.get("logging", "backup_count", default=5)

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

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)

    has_file_handler = any(
        isinstance(h, logging.handlers.RotatingFileHandler) and getattr(h, "baseFilename", None) == str(log_file)
        for h in root.handlers
    )
    if not has_file_handler:
        root.addHandler(fh)

    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.handlers.RotatingFileHandler)
        for h in root.handlers
    )
    if not has_stream_handler:
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
            pinch_threshold_ratio=config.get("gesture", "pinch_start_threshold", default=0.35),
        )

        # ── Phase 2: shadow-mode recognition ──────────────────────────
        logger.info("Model: RuleEngine (geometry) — 5 gestures")
        self._static_recognizer = StaticClassifier(self.engine)
        self._onnx_recognizer: GestureRecognizer | None = None
        self._shadow_threshold = config.get("system", "shadow_confidence_threshold", default=0.90)

        onnx_path = Path(__file__).parent / "models" / "classifier.onnx"
        if onnx_path.exists():
            try:
                from .pipeline.classifier import ONNXClassifier
                self._onnx_recognizer = ONNXClassifier(onnx_path)
                logger.info("Model: ONNX MLP classifier loaded — shadow threshold=%.0f%%",
                            self._shadow_threshold * 100)
            except Exception:
                logger.warning("Model: ONNX load failed, using rules only", exc_info=True)
        else:
            logger.info("Model: ONNX not found at %s — rules only", onnx_path)

        self.tap = AirTapDetector(
            tap_threshold=config.get("gesture", "tap_threshold", default=0.3),
            min_bend=config.get("gesture", "tap_min_bend", default=0.15),
            suppress_threshold=config.get("gesture", "tap_suppress_threshold", default=0.1),
            bend_timeout=config.get("gesture", "tap_bend_timeout", default=12),
            rebound_timeout=config.get("gesture", "tap_rebound_timeout", default=12),
            cooldown_frames=config.get("gesture", "tap_cooldown_frames", default=8),
        )
        self._tap_result: TapResult | None = None
        self.pinch = PinchDetector(
            pinch_threshold_ratio=config.get("gesture", "pinch_start_threshold", default=0.35),
            release_threshold_ratio=config.get("gesture", "pinch_release_threshold", default=0.55),
        )
        try:
            # Virtual desktop = all monitors combined
            sw = ctypes.windll.user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
            sh = ctypes.windll.user32.GetSystemMetrics(79)  # SM_CYVIRTUALSCREEN
        except Exception:
            sw, sh = 1920, 1080
            logger.info("Fallback screen dimensions used: %dx%d", sw, sh)
        self.cursor = CursorController(
            screen_width=sw, screen_height=sh,
            sensitivity=config.get("cursor", "sensitivity", default=0.8),
            deadzone=config.get("cursor", "deadzone", default=0.003),
            tap_deadzone=config.get("cursor", "tap_deadzone", default=0.012),
            beta=config.get("cursor", "smoothing_beta", default=0.007),
            fcmin=config.get("cursor", "smoothing_fcmin", default=1.0),
            min_cutoff=config.get("cursor", "smoothing_cutoff", default=1.0),
        )
        self.scroll = ScrollDetector(
            screen_height=sh,
            sensitivity=config.get("scroll", "sensitivity", default=2.0),
            deadzone=config.get("scroll", "deadzone", default=0.03),
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
        self._shadow_frame = 0

        # ── Diagnostics ────────────────────────────────────────────────
        self._fps = 0.0
        self._fps_timer = time.time()
        self._fps_frames = 0
        self._inference_source = "rule"
        self._onnx_conf = 0.0

        # ── Phase 3: DTW custom gesture matching ───────────────────────
        self._dtw_matcher = None
        self._dtw_trainer = None
        self._trainer_requested = False
        self._trainer_label = ""
        self._trainer_action: dict = {}
        self._trainer_take = 0
        self._dtw_cooldown = 0
        self._dtw_status: str = ""
        try:
            from .recognition.dtw_matcher import DtwMatcher
            from .recognition.dtw_trainer import DtwTrainer
            self._dtw_matcher = DtwMatcher(config)
            self._dtw_trainer = DtwTrainer(config)
            logger.info("Model: DTW matcher loaded — %d templates, threshold=%.1f",
                        self._dtw_matcher.template_count,
                        config.get("dtw", "match_threshold", default=8.0))
        except Exception:
            logger.warning("Model: DTW not available (fastdtw missing?)", exc_info=True)

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

            # Gesture label (Chinese)
            cn_label = _GESTURE_CN.get(gesture.name, gesture.name)
            color = (0, 255, 0)
            if gesture == Gesture.FIST:
                color = (0, 0, 255)
            elif gesture == Gesture.TWO_FINGER:
                color = (255, 0, 0)
            elif gesture == Gesture.PINCH:
                color = (255, 255, 0)
            label = f"{cn_label} | 拖拽={'是' if self.input_ctrl.is_dragging else '否'}"
            display = _cjk_text(display, label, (10, 30), 28, color)

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

            # ── Diagnostic panel (top-right) ──────────────────────────
            dz_mode = "TAP" if self.cursor._tap_active else "NORM"
            dz_color = (0, 255, 255) if dz_mode == "TAP" else (100, 200, 100)
            diag_lines = [
                f"FPS:{self._fps:.0f}",
                f"DZ:{dz_mode}",
                f"{self._inference_source.upper()}:{self._onnx_conf:.2f}" if self._onnx_recognizer else "RULE",
            ]
            y_off = 50
            for line in diag_lines:
                display = _cjk_text(display, line, (w - 10, y_off), 16,
                                    dz_color, anchor="ra")
                y_off += 22

            # DTW state overlay
            if self._dtw_matcher is not None:
                state_name = self._dtw_matcher.state.name
                status = self._dtw_status if self._dtw_status else state_name
                display = _cjk_text(display, f"DTW: {status}", (w - 10, 10), 18,
                                    (0, 255, 255), anchor="ra")
                if self._dtw_matcher.template_count > 0:
                    cv2.putText(display, f"{self._dtw_matcher.template_count} templates",
                                (w - 10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (200, 200, 100), 1)
        else:
            display = _cjk_text(display, "无手", (10, 30), 32, (0, 0, 255))

        cv2.imshow("MicroGesture Preview", display)
        cv2.waitKey(1)

    def _process_frame(self) -> None:
        # ── FPS counter ────────────────────────────────────────────────
        self._fps_frames += 1
        now = time.time()
        elapsed = now - self._fps_timer
        if elapsed >= 1.0:
            self._fps = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_timer = now

        frame = self.capture.latest_frame()
        if frame is None:
            return

        hand = self.detector.detect(frame)
        if hand is None:
            self._handle_no_hand()
            self._draw_preview(frame, None, None)
            return

        if self._no_hand_start is not None:
            lost_time = time.time() - self._no_hand_start
            if lost_time > 1.0:
                logger.info("Hand re-detected after %.1fs", lost_time)
        self._no_hand_start = None
        self._tap_result = self.tap.update(hand.landmarks)

        # ── Phase 2: shadow recognition ───────────────────────────────
        rule_result = self._static_recognizer.predict(hand.landmarks)
        gesture = Gesture[rule_result.label]

        # ONNX runs every frame
        self._shadow_frame += 1
        model_source = "rule"
        onnx_conf = 0.0
        onnx_agreed = True
        if self._onnx_recognizer is not None:
            onnx_result = self._onnx_recognizer.predict(hand.landmarks)
            onnx_conf = onnx_result.confidence
            if onnx_result.confidence >= self._shadow_threshold:
                model_source = "onnx"
                if Gesture[onnx_result.label] != gesture:
                    onnx_agreed = False
                gesture = Gesture[onnx_result.label]
        self._inference_source = model_source
        self._onnx_conf = onnx_conf

        # ── Periodic model inference summary (every 150 frames ≈ 5s) ──
        if self._shadow_frame % 150 == 0:
            parts = [
                f"gesture={gesture.name}",
                f"source={model_source}",
            ]
            if self._onnx_recognizer is not None:
                parts.append(f"onnx_conf={onnx_conf:.2f}")
                if not onnx_agreed:
                    parts.append(f"rule_said={rule_result.label}")
            logger.debug("Inference: %s", " | ".join(parts))

        # Handle gesture transitions
        if gesture == Gesture.PALM_OPEN or gesture == Gesture.PINCH:
            self._fist_start = None

        # ── Phase 3: DTW training mode ──────────────────────────────────
        if self._trainer_requested and self._dtw_trainer is not None:
            take_num = self._dtw_trainer.feed(hand.landmarks, gesture)
            if take_num is not None:
                self._trainer_take = take_num
                self._dtw_status = f"训练: 第{take_num}/3次"
                logger.info("DTW training: take %d/3 recorded", take_num)
                if take_num >= 3:
                    result = self._dtw_trainer.finish()
                    if result and self._dtw_matcher:
                        self._dtw_matcher.add_template(
                            result.name, result.label, result.sequence,
                            self._trainer_action,
                        )
                        self._dtw_status = f"已注册: {result.label}"
                        logger.info("DTW template registered: %s (action=%s)",
                                    result.name, self._trainer_action)
                    self._trainer_requested = False
            self._draw_preview(frame, hand, gesture)
            return

        # ── Phase 3: DTW matching (parallel path, before normal dispatch) ──
        dtw_match = None
        if self._dtw_matcher is not None:
            dtw_match = self._dtw_matcher.feed(hand.landmarks, gesture)
            if dtw_match is not None:
                self._dtw_status = f"匹配: {dtw_match.label} ({dtw_match.confidence:.0%})"
                logger.info("DTW match: %s (dist=%.2f conf=%.2f)",
                            dtw_match.name, dtw_match.distance, dtw_match.confidence)
                self._execute_dtw_action(dtw_match)

        # Dispatch by gesture
        if gesture == Gesture.PALM_OPEN:
            self.cursor.unfreeze()
            self._handle_cursor_move(hand.landmarks)
            self._handle_tap()
            self._handle_pinch(hand.landmarks)

        elif gesture == Gesture.FIST:
            self.cursor.freeze()
            if self._fist_start is None:
                self._fist_start = time.time()
            if self._right_click_mode == "fist_tap":
                self._handle_tap(right_click=True)

        elif gesture == Gesture.TWO_FINGER:
            self.cursor.freeze()
            self._handle_scroll(hand.landmarks)
            if self._right_click_mode == "two_finger":
                self._handle_tap(right_click=True)

        elif gesture == Gesture.PINCH:
            self.cursor.unfreeze()
            self._handle_cursor_move(hand.landmarks)
            self._handle_pinch(hand.landmarks)

        if self._prev_gesture == Gesture.TWO_FINGER and gesture != Gesture.TWO_FINGER:
            self.scroll.stop()

        self._draw_preview(frame, hand, gesture)
        self._prev_gesture = gesture

    def _handle_no_hand(self) -> None:
        timeout = self.config.get("system", "no_hand_sleep_timeout", default=5.0)
        if self._no_hand_start is None:
            self._no_hand_start = time.time()
        elif timeout > 0 and (time.time() - self._no_hand_start) > timeout:
            self.cursor.freeze()
            self.scroll.stop()

    def _handle_cursor_move(self, landmarks) -> None:
        # Full cursor suppression when finger is bending hard
        if self._tap_result is not None and self._tap_result.suppress_cursor:
            if self._shadow_frame % 5 == 0:
                logger.debug("光标: 点按抑制 弯曲比=%.3f", self._tap_result.dratio)
            return
        # Larger deadzone when tracking a potential tap
        self.cursor._tap_active = self.tap.is_tapping
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

    def _execute_dtw_action(self, match) -> None:
        """Execute the action configured for a matched DTW gesture."""
        action = getattr(match, "action_config", {})
        atype = action.get("type", "key_combo")
        if atype == "key_combo":
            modifiers = action.get("modifiers", [])
            key = action.get("key")
            if key:
                self.input_ctrl.key_combo(modifiers, key)
                logger.info("DTW action: %s+%s", "+".join(modifiers + [key]) if modifiers else key)
        self._dtw_cooldown = self.config.get("dtw", "cooldown_frames", default=90)

    def handle_register_gesture(self, name: str = "", label: str = "",
                                 action_config: dict | None = None) -> None:
        """Initiate gesture registration. Called from tray callback."""
        if self._dtw_trainer is None:
            logger.warning("DTW trainer not available")
            return
        if self._trainer_requested:
            logger.warning("Training already in progress")
            return
        if action_config is None:
            action_config = {"type": "key_combo", "modifiers": ["ctrl"], "key": ""}
        if not name:
            import time
            name = f"gesture_{time.strftime('%Y%m%d_%H%M%S')}"
        if not label:
            label = name
        self._trainer_requested = True
        self._trainer_label = label
        self._trainer_action = action_config
        self._trainer_take = 0
        self._dtw_status = "训练: 握拳开始录制..."
        self._dtw_trainer.start(name, label)
        logger.info("DTW training started: name=%s label=%s action=%s",
                    name, label, action_config)

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

    def set_scroll_sensitivity(self, value: float) -> None:
        self.scroll.sensitivity = value
        logger.info("Scroll sensitivity set to %.1f", value)

    def set_scroll_deadzone(self, value: float) -> None:
        self.scroll._deadzone = value
        logger.info("Scroll deadzone set to %.3f", value)

    def set_shadow_threshold(self, value: float) -> None:
        self._shadow_threshold = value
        logger.info("Shadow threshold set to %.0f%%", value * 100)

    def open_settings(self) -> None:
        """Open the settings GUI panel."""
        try:
            from .system.settings_gui import open_settings_panel
            panel = open_settings_panel(
                cursor_sensitivity=self.cursor.sensitivity,
                scroll_sensitivity=self.scroll.sensitivity,
                scroll_deadzone=self.scroll._deadzone,
                right_click_mode=self._right_click_mode,
                shadow_threshold=self._shadow_threshold,
                on_cursor_sensitivity=self.set_sensitivity,
                on_scroll_sensitivity=self.set_scroll_sensitivity,
                on_scroll_deadzone=self.set_scroll_deadzone,
                on_right_click=self.set_right_click,
                on_shadow_threshold=self.set_shadow_threshold,
            )
            logger.info("Settings panel opened")
        except Exception:
            logger.exception("Failed to open settings panel")

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MicroGesture hand gesture app")
    parser.add_argument("--config", type=Path, help="Path to an alternate config file")
    parser.add_argument("--no-watch", action="store_true", help="Disable config file watcher")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config = get_config(args.config, watch=not args.no_watch)
    setup_logging(config, debug=args.debug)

    logger.info("MicroGesture v0.1.0 starting")

    pipeline = GesturePipeline(config)

    tray = SystemTray(TrayCallbacks(
        toggle_tracking=pipeline.toggle_tracking,
        set_sensitivity=pipeline.set_sensitivity,
        set_right_click=pipeline.set_right_click,
        quit_app=lambda: shutdown(pipeline, tray),
        register_gesture=lambda: pipeline.handle_register_gesture(),
        open_settings=lambda: pipeline.open_settings(),
    ))

    pipeline.start()

    def _tray_thread():
        # Bind status callback AFTER tray.run() initialises the icon
        pipeline._on_status_change = lambda c: tray.set_state(
            TrayState.NORMAL if c else TrayState.NO_CAMERA
        )
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
