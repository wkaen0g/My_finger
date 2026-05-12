"""System tray icon and menu for microgesture control."""

import logging
import threading
from enum import Enum, auto
from typing import Callable, Optional

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


class TrayState(Enum):
    NORMAL = auto()
    NO_CAMERA = auto()
    SLEEP = auto()


def _make_icon(color: str) -> Image.Image:
    img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([4, 4, 28, 28], fill=color, outline="white", width=2)
    return img


_ICONS = {
    TrayState.NORMAL: _make_icon("green"),
    TrayState.NO_CAMERA: _make_icon("orange"),
    TrayState.SLEEP: _make_icon("gray"),
}


class SystemTray:
    """pystray system tray with menu for gesture control."""

    def __init__(self, callbacks: dict):
        """
        callbacks keys: toggle_tracking, set_sensitivity, set_right_click, quit_app
        """
        self._callbacks = callbacks
        self._icon = None
        self._state = TrayState.NORMAL
        self._running = False

    def _build_menu(self):
        from pystray import Menu, MenuItem

        def _toggle(icon, item):
            self._callbacks.get("toggle_tracking", lambda: None)()

        def _sensitivity_low(icon, item):
            self._callbacks.get("set_sensitivity", lambda v: None)(1.2)

        def _sensitivity_med(icon, item):
            self._callbacks.get("set_sensitivity", lambda v: None)(2.4)

        def _sensitivity_high(icon, item):
            self._callbacks.get("set_sensitivity", lambda v: None)(3.6)

        def _rc_fist(icon, item):
            self._callbacks.get("set_right_click", lambda v: None)("fist_tap")

        def _rc_two_finger(icon, item):
            self._callbacks.get("set_right_click", lambda v: None)("two_finger")

        def _quit(icon, item):
            self._icon.stop() if self._icon else None
            self._callbacks.get("quit_app", lambda: None)()

        return Menu(
            MenuItem("Toggle Tracking", _toggle, default=True),
            Menu.SEPARATOR,
            MenuItem("Sensitivity", Menu(
                MenuItem("Low", _sensitivity_low, checked=lambda item: False),
                MenuItem("Medium", _sensitivity_med, checked=lambda item: True),
                MenuItem("High", _sensitivity_high, checked=lambda item: False),
            )),
            MenuItem("Right Click Mode", Menu(
                MenuItem("Fist + Tap", _rc_fist, checked=lambda item: True),
                MenuItem("Two Finger Tap", _rc_two_finger, checked=lambda item: False),
            )),
            Menu.SEPARATOR,
            MenuItem("Quit", _quit),
        )

    def _on_quit_callback(self):
        self._running = False

    def set_state(self, state: TrayState) -> None:
        self._state = state
        if self._icon:
            self._icon.icon = _ICONS[state]

    def run(self) -> None:
        import pystray

        self._icon = pystray.Icon(
            "microgesture",
            _ICONS[self._state],
            "MicroGesture",
            self._build_menu(),
        )
        self._running = True
        self._icon.run()

    def stop(self) -> None:
        self._running = False
        if self._icon:
            self._icon.stop()
