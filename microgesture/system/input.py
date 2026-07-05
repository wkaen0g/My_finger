"""System input simulation via Windows native API (fast path, no pyautogui)."""

import ctypes
import logging
from ctypes import wintypes

logger = logging.getLogger(__name__)

# ── Win32 API bindings ────────────────────────────────────────────────────

_user32 = ctypes.windll.user32

# Screen metrics
_SM_CXSCREEN = 0
_SM_CYSCREEN = 1

# Mouse event flags
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_RIGHTDOWN = 0x0008
_MOUSEEVENTF_RIGHTUP = 0x0010
_MOUSEEVENTF_MIDDLEDOWN = 0x0020
_MOUSEEVENTF_MIDDLEUP = 0x0040
_MOUSEEVENTF_WHEEL = 0x0800
_MOUSEEVENTF_ABSOLUTE = 0x8000

_WHEEL_DELTA = 120


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _mouse_event(dw_flags: int, dx: int = 0, dy: int = 0,
                 dw_data: int = 0) -> None:
    _user32.mouse_event(dw_flags, dx, dy, dw_data, 0)


def _get_cursor_pos() -> tuple[int, int]:
    pt = _POINT()
    _user32.GetCursorPos(ctypes.byref(pt))
    return (pt.x, pt.y)

# ── input controller ──────────────────────────────────────────────────────

class InputController:
    """Unified interface for simulating mouse events via Win32 API."""

    def __init__(self):
        self._dragging = False
        self._screen_w = _user32.GetSystemMetrics(_SM_CXSCREEN)
        self._screen_h = _user32.GetSystemMetrics(_SM_CYSCREEN)

    # ── cursor ────────────────────────────────────────────────────────

    def move(self, dx: float, dy: float) -> None:
        """Move cursor by relative delta. Uses SetCursorPos (fast)."""
        if dx == 0 and dy == 0:
            return
        idx = int(round(dx))
        idy = int(round(dy))
        x, y = _get_cursor_pos()
        nx, ny = x + idx, y + idy

        # Soft boundary: clamp to screen, log if near edge
        nx = max(0, min(self._screen_w - 1, nx))
        ny = max(0, min(self._screen_h - 1, ny))

        _user32.SetCursorPos(nx, ny)

    # ── clicks ─────────────────────────────────────────────────────────

    def click(self) -> None:
        if self._dragging:
            return
        _mouse_event(_MOUSEEVENTF_LEFTDOWN)
        _mouse_event(_MOUSEEVENTF_LEFTUP)
        logger.debug("Click")

    def double_click(self) -> None:
        if self._dragging:
            return
        self.click()
        self.click()
        logger.debug("Double click")

    def right_click(self) -> None:
        if self._dragging:
            return
        _mouse_event(_MOUSEEVENTF_RIGHTDOWN)
        _mouse_event(_MOUSEEVENTF_RIGHTUP)
        logger.debug("Right click")

    # ── drag ───────────────────────────────────────────────────────────

    def drag_start(self) -> None:
        if self._dragging:
            return
        _mouse_event(_MOUSEEVENTF_LEFTDOWN)
        self._dragging = True
        logger.debug("Drag start")

    def drag_end(self) -> None:
        if not self._dragging:
            return
        _mouse_event(_MOUSEEVENTF_LEFTUP)
        self._dragging = False
        logger.debug("Drag end")

    # ── scroll ─────────────────────────────────────────────────────────

    def scroll(self, amount: int) -> None:
        """Scroll by amount (positive=up, negative=down)."""
        _mouse_event(_MOUSEEVENTF_WHEEL, dw_data=int(round(amount)) * _WHEEL_DELTA)

    @property
    def is_dragging(self) -> bool:
        return self._dragging
