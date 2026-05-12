"""System input simulation via PyAutoGUI."""

import logging
import time
from typing import Tuple

import pyautogui

logger = logging.getLogger(__name__)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0


class InputController:
    """Unified interface for simulating mouse and keyboard events."""

    def __init__(self):
        self._dragging = False
        self._screen_w, self._screen_h = pyautogui.size()

    def move(self, dx: float, dy: float) -> None:
        """Move cursor by relative delta."""
        if dx == 0 and dy == 0:
            return
        pyautogui.moveRel(int(round(dx)), int(round(dy)), _pause=False)

    def click(self) -> None:
        if self._dragging:
            return
        pyautogui.click(_pause=False)
        logger.debug("Click")

    def double_click(self) -> None:
        if self._dragging:
            return
        pyautogui.doubleClick(_pause=False)
        logger.debug("Double click")

    def right_click(self) -> None:
        if self._dragging:
            return
        pyautogui.rightClick(_pause=False)
        logger.debug("Right click")

    def drag_start(self) -> None:
        if self._dragging:
            return
        pyautogui.mouseDown(_pause=False)
        self._dragging = True
        logger.debug("Drag start")

    def drag_end(self) -> None:
        if not self._dragging:
            return
        pyautogui.mouseUp(_pause=False)
        self._dragging = False
        logger.debug("Drag end")

    def scroll(self, amount: int) -> None:
        """Scroll by amount (positive=up, negative=down)."""
        pyautogui.scroll(int(round(amount)), _pause=False)

    @property
    def is_dragging(self) -> bool:
        return self._dragging
