import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from microgesture.config import Config
from microgesture.main import GesturePipeline, _cjk_text, setup_logging


class DummyCapture:
    def __init__(self, *args, **kwargs):
        self.is_connected = False

    def latest_frame(self):
        return None

    def start(self):
        pass

    def stop(self):
        pass


class DummyDetector:
    def __init__(self, *args, **kwargs):
        pass

    def detect(self, frame):
        return None

    def close(self):
        pass


class DummyRuleEngine:
    def __init__(self, *args, **kwargs):
        pass


class DummyStaticClassifier:
    def __init__(self, engine):
        pass

    def predict(self, landmarks):
        return SimpleNamespace(label="NO_HAND")


class DummyAirTap:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, landmarks):
        return None


class DummyPinch:
    def __init__(self, *args, **kwargs):
        pass


class DummyCursor:
    def __init__(self, screen_width, screen_height, *args, **kwargs):
        self.screen_width = screen_width
        self.screen_height = screen_height

    def update(self, x, y):
        return 0.0, 0.0

    def freeze(self):
        pass

    def unfreeze(self):
        pass


class DummyScroll:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def update(self, a, b):
        return 0.0


class DummyInput:
    def __init__(self):
        self.is_dragging = False

    def move(self, dx, dy):
        pass

    def click(self):
        pass

    def right_click(self):
        pass

    def drag_start(self):
        pass

    def drag_end(self):
        pass


def test_cjk_text_returns_same_shape():
    image = np.zeros((80, 160, 3), dtype=np.uint8)
    output = _cjk_text(image, "测试", (10, 10), font_size=14, color=(0, 255, 0))
    assert output.shape == image.shape
    assert output.dtype == image.dtype


def test_setup_logging_creates_handlers_once(tmp_path: Path):
    class DummyConfig:
        def get(self, section, name, default=None):
            if section == "logging":
                return {
                    "dir": str(tmp_path),
                    "level": "DEBUG",
                    "max_bytes": 1024,
                    "backup_count": 1,
                }.get(name, default)
            return default

    root = logging.getLogger()
    original_handlers = root.handlers[:]
    root.handlers.clear()
    try:
        setup_logging(DummyConfig())
        initial_count = len(root.handlers)
        assert initial_count >= 2

        setup_logging(DummyConfig())
        assert len(root.handlers) == initial_count
    finally:
        root.handlers[:] = original_handlers


def test_gesture_pipeline_fallback_screen_dimensions(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({}), encoding="utf-8")
    config = Config(path=config_path, watch=False)

    with patch("microgesture.main.CameraCapture", DummyCapture), \
         patch("microgesture.main.HandDetector", DummyDetector), \
         patch("microgesture.main.RuleEngine", DummyRuleEngine), \
         patch("microgesture.main.StaticClassifier", DummyStaticClassifier), \
         patch("microgesture.main.AirTapDetector", DummyAirTap), \
         patch("microgesture.main.PinchDetector", DummyPinch), \
         patch("microgesture.main.CursorController", DummyCursor), \
         patch("microgesture.main.ScrollDetector", DummyScroll), \
         patch("microgesture.main.InputController", DummyInput), \
         patch("microgesture.main.ctypes.windll", None):
        pipeline = GesturePipeline(config)

    assert pipeline.cursor.screen_width == 1920
    assert pipeline.cursor.screen_height == 1080
