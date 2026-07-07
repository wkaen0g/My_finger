"""System input simulation via Windows native API (fast path, no pyautogui)."""

import ctypes
import logging
from ctypes import wintypes

logger = logging.getLogger(__name__)

# ── Win32 API bindings ────────────────────────────────────────────────────

_user32 = ctypes.windll.user32

# Screen metrics (virtual desktop = all monitors combined)
_SM_XVIRTUALSCREEN = 76
_SM_YVIRTUALSCREEN = 77
_SM_CXVIRTUALSCREEN = 78
_SM_CYVIRTUALSCREEN = 79

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

# Keyboard event flags
_KEYEVENTF_EXTENDEDKEY = 0x0001
_KEYEVENTF_KEYUP = 0x0002

_MODIFIER_VK = {
    "ctrl": 0x11,
    "alt": 0x12,
    "shift": 0x10,
    "win": 0x5B,
}

_KEY_VK = {
    "a": 0x41, "b": 0x42, "c": 0x43, "d": 0x44, "e": 0x45,
    "f": 0x46, "g": 0x47, "h": 0x48, "i": 0x49, "j": 0x4A,
    "k": 0x4B, "l": 0x4C, "m": 0x4D, "n": 0x4E, "o": 0x4F,
    "p": 0x50, "q": 0x51, "r": 0x52, "s": 0x53, "t": 0x54,
    "u": 0x55, "v": 0x56, "w": 0x57, "x": 0x58, "y": 0x59, "z": 0x5A,
    "0": 0x30, "1": 0x31, "2": 0x32, "3": 0x33, "4": 0x34,
    "5": 0x35, "6": 0x36, "7": 0x37, "8": 0x38, "9": 0x39,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
    "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77,
    "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
    "left": 0x25, "right": 0x27, "up": 0x26, "down": 0x28,
    "enter": 0x0D, "space": 0x20, "tab": 0x09,
    "escape": 0x1B, "backspace": 0x08, "delete": 0x2E,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "printscreen": 0x2C, "volume_mute": 0xAD,
    "volume_down": 0xAE, "volume_up": 0xAF,
    "media_next": 0xB0, "media_prev": 0xB1, "media_play": 0xB3,
}

def _keybd_event(vk_code: int, up: bool = False) -> None:
    flags = _KEYEVENTF_KEYUP if up else 0
    if vk_code in (0x5B, 0x5C):  # WIN keys need extended flag
        flags |= _KEYEVENTF_EXTENDEDKEY
    _user32.keybd_event(vk_code, 0, flags, 0)


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
        # Virtual desktop = bounding rect of all monitors
        self._screen_x = _user32.GetSystemMetrics(_SM_XVIRTUALSCREEN)
        self._screen_y = _user32.GetSystemMetrics(_SM_YVIRTUALSCREEN)
        self._screen_w = _user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN)
        self._screen_h = _user32.GetSystemMetrics(_SM_CYVIRTUALSCREEN)

    # ── cursor ────────────────────────────────────────────────────────

    def move(self, dx: float, dy: float) -> None:
        """Move cursor by relative delta. Uses SetCursorPos (fast)."""
        if dx == 0 and dy == 0:
            return
        idx = int(round(dx))
        idy = int(round(dy))
        x, y = _get_cursor_pos()
        nx, ny = x + idx, y + idy

        # Soft boundary: clamp to virtual desktop
        nx = max(self._screen_x, min(self._screen_x + self._screen_w - 1, nx))
        ny = max(self._screen_y, min(self._screen_y + self._screen_h - 1, ny))

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

    # ── keyboard ─────────────────────────────────────────────────────────

    def key_combo(self, modifiers: list[str], key: str) -> None:
        """Simulate a keyboard shortcut like Ctrl+C or Win+D.

        Args:
            modifiers: List of modifier keys (e.g. ['ctrl', 'shift'])
            key: The main key (e.g. 'c', 'enter', 'f5')
        """
        vk_key = _KEY_VK.get(key.lower())
        if vk_key is None:
            logger.warning("Unknown key: %s", key)
            return

        # Press modifiers
        for mod in modifiers:
            vk_mod = _MODIFIER_VK.get(mod.lower())
            if vk_mod is None:
                logger.warning("Unknown modifier: %s", mod)
                continue
            _keybd_event(vk_mod, up=False)

        # Press key
        _keybd_event(vk_key, up=False)

        # Small delay for reliability
        import time
        time.sleep(0.03)

        # Release key
        _keybd_event(vk_key, up=True)

        # Release modifiers in reverse order
        for mod in reversed(modifiers):
            vk_mod = _MODIFIER_VK.get(mod.lower())
            if vk_mod is not None:
                _keybd_event(vk_mod, up=True)
