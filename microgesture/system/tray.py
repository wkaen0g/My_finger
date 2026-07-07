"""System tray icon and menu for microgesture control."""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Callable

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# ── icon drawing ──────────────────────────────────────────────────────────

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

# ── callbacks type ─────────────────────────────────────────────────────────

class TrayCallbacks:
    """Typed callbacks the tray expects from the pipeline / main."""

    def __init__(
        self,
        toggle_tracking: Callable[[], None],
        set_sensitivity: Callable[[float], None],
        set_right_click: Callable[[str], None],
        quit_app: Callable[[], None],
        register_gesture: Callable[[], None] | None = None,
        open_settings: Callable[[], None] | None = None,
        manage_gestures: Callable[[], None] | None = None,
    ):
        self.toggle_tracking = toggle_tracking
        self.set_sensitivity = set_sensitivity
        self.set_right_click = set_right_click
        self.quit_app = quit_app
        self.register_gesture = register_gesture
        self.open_settings = open_settings
        self.manage_gestures = manage_gestures


# ── tray ───────────────────────────────────────────────────────────────────

class SystemTray:
    """pystray system tray with menu for gesture control."""

    _SENSITIVITY_MAP = {"low": 0.5, "medium": 0.8, "high": 1.6}

    def __init__(self, callbacks: TrayCallbacks):
        self._cb = callbacks
        self._icon = None
        self._state = TrayState.NORMAL
        self._sensitivity_level = "medium"
        self._right_click_mode = "fist_tap"
        self._tracking_enabled = True

    # ── state mutators (call pipeline + refresh menu) ──────────────────

    def _refresh(self) -> None:
        if self._icon:
            self._icon.update_menu()

    def _on_toggle_tracking(self) -> None:
        self._tracking_enabled = not self._tracking_enabled
        self._cb.toggle_tracking()
        self._refresh()

    def _on_sensitivity(self, level: str) -> None:
        self._sensitivity_level = level
        self._cb.set_sensitivity(self._SENSITIVITY_MAP[level])
        self._refresh()

    def _on_right_click_mode(self, mode: str) -> None:
        self._right_click_mode = mode
        self._cb.set_right_click(mode)
        self._refresh()

    def _on_quit(self) -> None:
        self._cb.quit_app()

    def _on_register_gesture(self) -> None:
        if self._cb.register_gesture:
            self._cb.register_gesture()

    def _on_open_settings(self) -> None:
        if self._cb.open_settings:
            self._cb.open_settings()

    def _on_manage_gestures(self) -> None:
        if self._cb.manage_gestures:
            self._cb.manage_gestures()

    # ── menu ───────────────────────────────────────────────────────────

    def _build_menu(self):
        from pystray import Menu, MenuItem

        sens_menu = Menu(
            MenuItem("Low", lambda i: self._on_sensitivity("low"),
                     checked=lambda i: self._sensitivity_level == "low"),
            MenuItem("Medium", lambda i: self._on_sensitivity("medium"),
                     checked=lambda i: self._sensitivity_level == "medium"),
            MenuItem("High", lambda i: self._on_sensitivity("high"),
                     checked=lambda i: self._sensitivity_level == "high"),
        )

        rc_menu = Menu(
            MenuItem("Fist + Tap", lambda i: self._on_right_click_mode("fist_tap"),
                     checked=lambda i: self._right_click_mode == "fist_tap"),
            MenuItem("Two Finger", lambda i: self._on_right_click_mode("two_finger"),
                     checked=lambda i: self._right_click_mode == "two_finger"),
        )

        return Menu(
            MenuItem("Toggle Tracking", lambda i: self._on_toggle_tracking(),
                     checked=lambda i: self._tracking_enabled),
            Menu.SEPARATOR,
            MenuItem("Sensitivity", sens_menu),
            MenuItem("Right Click", rc_menu),
            Menu.SEPARATOR,
            MenuItem("Register Gesture ...", lambda i: self._on_register_gesture(),
                     enabled=lambda i: self._cb.register_gesture is not None),
            MenuItem("Manage Gestures ...", lambda i: self._on_manage_gestures(),
                     enabled=lambda i: self._cb.manage_gestures is not None),
            MenuItem("Settings ...", lambda i: self._on_open_settings(),
                     enabled=lambda i: self._cb.open_settings is not None),
            Menu.SEPARATOR,
            MenuItem("Quit", lambda i: self._on_quit()),
        )

    # ── lifecycle ──────────────────────────────────────────────────────

    def set_state(self, state: TrayState) -> None:
        self._state = state
        if self._icon is not None:
            try:
                self._icon.icon = _ICONS[state]
            except Exception:
                logger.debug("Tray icon update deferred (not ready yet)")

    def run(self) -> None:
        import pystray

        self._icon = pystray.Icon(
            "microgesture",
            _ICONS[self._state],
            "MicroGesture",
            self._build_menu(),
        )
        self._icon.run()

    def stop(self) -> None:
        if self._icon:
            self._icon.stop()
            self._icon = None
